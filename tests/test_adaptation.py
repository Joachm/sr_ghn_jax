from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from adaptation_analysis import comparison_label
from configs import (
    BASELINE_FIXED_LR,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FULL,
    BASELINE_NO_SELF_REFERENCE,
    GYMNAX_SUITE_DEFAULT_EVALS_PER_GENERATION,
    GYMNAX_SUITE_DEFAULT_NUM_GENERATIONS,
    GYMNAX_SUITE_DEFAULT_POP_SIZE,
    gymnax_control_pop_size,
    ShiftWindowConfig,
    make_config_gymnax_generic,
    make_config_nonstationary_brax,
    make_config_nonstationary_gymnax,
)
from evosax_adapter import available_evosax_algorithms, fitness_for_evosax, resolve_evosax_strategy
from envs import (
    active_shift_mask,
    apply_reward_shifts,
    iter_shift_windows,
    make_env,
    map_action_for_shifts,
    map_observation_for_shifts,
)
from experiments._adaptation import (
    GYMNAX_MINATAR_SUITE_ENVIRONMENTS,
    gymnax_minatar_suite_shift_windows,
    gymnax_suite_shift_windows,
)
from experiments._common import build_graphs_and_specs, default_wandb_project, run_experiment, wandb_config_payload
import evolution as evolution_module
from evolution import evo_step, init_population, EvoState
from meta_sine_srghn import MetaSineConfig, parse_condition_spec, run_condition
from obs_norm import flatten_observation, init_obs_norm, normalize_obs
from policy import apply_policy
from policy_vectors import flatten_policy_params, policy_num_dims, unflatten_policy_vector
from srghn import mutate_with_metadata
from specs import _infer_obs_dim as infer_specs_obs_dim
from specs import policy_spec_for_task


class AdaptationTests(unittest.TestCase):
    def _make_minatar_config(self, env_id: str = "Asterix-MinAtar", **kwargs):
        defaults = {
            "policy_architecture": "cnn_mlp",
            "policy_conv_channels": (16,),
            "policy_conv_kernel_sizes": ((3, 3),),
            "policy_conv_strides": ((1, 1),),
            "policy_hidden_dims": (128,),
        }
        defaults.update(kwargs)
        return make_config_nonstationary_gymnax(env_id, **defaults)

    def _make_meta_sine_config(self) -> MetaSineConfig:
        return MetaSineConfig(
            seed=0,
            outer_generations=1,
            meta_batch_size=2,
            test_task_batch_size=2,
            outer_pop_size=2,
            outer_children_per_parent=1,
            inner_pop_size=2,
            inner_children_per_parent=1,
            inner_generations=1,
            support_k=2,
            query_k=3,
            policy_hidden_dims=(8,),
            embedding_dim=8,
            gnn_hidden_dim=8,
            gnn_steps_policy=1,
            gnn_steps_self=1,
            stoch_coeff_dim=8,
            parameter_block_size=16,
            mutation_rate_head_dim=2,
        )

    def _assert_array_pytree_allclose(self, a, b):
        a_arr, _ = eqx.partition(a, eqx.is_array)
        b_arr, _ = eqx.partition(b, eqx.is_array)
        a_leaves = jax.tree_util.tree_leaves(a_arr)
        b_leaves = jax.tree_util.tree_leaves(b_arr)
        self.assertEqual(len(a_leaves), len(b_leaves))
        for left, right in zip(a_leaves, b_leaves):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))

    def test_shift_schedule_window_boundaries(self):
        config = make_config_nonstationary_gymnax(
            "CartPole-v1",
            shift_windows=(ShiftWindowConfig(3, 5, "cartpole_flip"),),
        )
        self.assertFalse(bool(active_shift_mask(jnp.asarray(2), config, "cartpole_flip")))
        self.assertTrue(bool(active_shift_mask(jnp.asarray(3), config, "cartpole_flip")))
        self.assertTrue(bool(active_shift_mask(jnp.asarray(5), config, "cartpole_flip")))
        self.assertFalse(bool(active_shift_mask(jnp.asarray(6), config, "cartpole_flip")))

    def test_action_and_reward_shift_rules(self):
        gymnax_config = make_config_nonstationary_gymnax(
            "CartPole-v1",
            shift_windows=(ShiftWindowConfig(1, 3, "cartpole_flip"),),
        )
        flipped = map_action_for_shifts(jnp.asarray(0), jnp.asarray(2), gymnax_config)
        self.assertEqual(int(flipped), 1)

        class DummyState:
            metrics = {"x_velocity": jnp.asarray(2.5, dtype=jnp.float32)}

        brax_config = make_config_nonstationary_brax(
            "ant",
            shift_windows=(ShiftWindowConfig(4, None, "brax_direction_switch"),),
        )
        shifted = apply_reward_shifts(jnp.asarray(1.0, dtype=jnp.float32), DummyState(), jnp.asarray(4), brax_config)
        self.assertAlmostEqual(float(shifted), -2.5, places=5)

        discrete_config = make_config_nonstationary_gymnax(
            "Acrobot-v1",
            shift_windows=(ShiftWindowConfig(1, 3, "discrete_reverse"),),
        )
        reversed_action = map_action_for_shifts(
            jnp.asarray(0),
            jnp.asarray(2),
            discrete_config,
            act_dim=3,
            is_discrete=True,
        )
        self.assertEqual(int(reversed_action), 2)

        continuous_config = make_config_nonstationary_gymnax(
            "MountainCarContinuous-v0",
            shift_windows=(ShiftWindowConfig(1, 3, "continuous_action_flip"),),
        )
        flipped_action = map_action_for_shifts(
            jnp.asarray([0.4], dtype=jnp.float32),
            jnp.asarray(2),
            continuous_config,
            act_dim=1,
            is_discrete=False,
        )
        self.assertAlmostEqual(float(flipped_action[0]), -0.4, places=5)

        pendulum_config = make_config_nonstationary_gymnax(
            "Pendulum-v1",
            shift_windows=(ShiftWindowConfig(1, 3, "pendulum_obs_flip"),),
        )
        obs = jnp.asarray([0.8, 0.2, 1.5], dtype=jnp.float32)
        shifted_obs = map_observation_for_shifts(obs, jnp.asarray(2), pendulum_config)
        self.assertAlmostEqual(float(shifted_obs[0]), 0.2, places=5)
        self.assertAlmostEqual(float(shifted_obs[1]), 0.8, places=5)
        self.assertAlmostEqual(float(shifted_obs[2]), -1.5, places=5)

        reward_passthrough = apply_reward_shifts(
            jnp.asarray(1.25, dtype=jnp.float32),
            DummyState(),
            jnp.asarray(2),
            discrete_config,
        )
        self.assertAlmostEqual(float(reward_passthrough), 1.25, places=5)

    def test_suite_windows_have_two_switches(self):
        for env_id in (
            "CartPole-v1",
            "Acrobot-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ):
            windows = gymnax_suite_shift_windows(env_id)
            self.assertEqual(len(windows), 2)

    def test_minatar_suite_windows_have_two_switches(self):
        for env_id in GYMNAX_MINATAR_SUITE_ENVIRONMENTS:
            windows = gymnax_minatar_suite_shift_windows(env_id)
            self.assertEqual(len(windows), 2)

    def test_minatar_config_uses_cnn_policy_architecture(self):
        config = self._make_minatar_config()
        self.assertEqual(config.policy_architecture, "cnn_mlp")
        self.assertEqual(config.policy_conv_channels, (16,))
        self.assertEqual(config.policy_conv_kernel_sizes, ((3, 3),))
        self.assertEqual(config.policy_conv_strides, ((1, 1),))
        self.assertEqual(config.policy_hidden_dims, (128,))

    def test_stationary_configs_stay_stationary(self):
        config = make_config_gymnax_generic("CartPole-v1")
        self.assertEqual(iter_shift_windows(config), ())
        action = map_action_for_shifts(jnp.asarray(1), jnp.asarray(100), config)
        self.assertEqual(int(action), 1)

    def test_evosax_defaults_match_suite_budget(self):
        srghn_config = make_config_nonstationary_gymnax("CartPole-v1")
        evosax_config = make_config_nonstationary_gymnax(
            "CartPole-v1",
            optimizer_family="evosax",
            baseline_name="evosax",
            evosax_algo="cma_es",
        )
        self.assertEqual(srghn_config.pop_size, GYMNAX_SUITE_DEFAULT_POP_SIZE)
        self.assertEqual(srghn_config.num_generations, GYMNAX_SUITE_DEFAULT_NUM_GENERATIONS)
        self.assertEqual(evosax_config.pop_size, GYMNAX_SUITE_DEFAULT_EVALS_PER_GENERATION)
        self.assertEqual(evosax_config.num_generations, GYMNAX_SUITE_DEFAULT_NUM_GENERATIONS)
        self.assertFalse(hasattr(evosax_config, "baseline_name"))
        self.assertFalse(hasattr(evosax_config, "parameter_block_size"))
        self.assertEqual(default_wandb_project(srghn_config), "srghn_jax")
        self.assertEqual(default_wandb_project(evosax_config), "sr-ghn_control_cma_es")

        shorter_branching = make_config_nonstationary_gymnax(
            "CartPole-v1",
            optimizer_family="evosax",
            baseline_name="evosax",
            evosax_algo="cma_es",
            children_per_parent=2,
        )
        self.assertEqual(shorter_branching.pop_size, gymnax_control_pop_size(2))

    def test_policy_vector_round_trip(self):
        try:
            config = make_config_nonstationary_gymnax("CartPole-v1")
            policy_spec = policy_spec_for_task(config)
            params = tuple(
                jnp.arange(size, dtype=jnp.float32).reshape(shape)
                for shape, size in zip(policy_spec.shapes, policy_spec.sizes)
            )
            vector = flatten_policy_params(params)
            restored = unflatten_policy_vector(vector, policy_spec)
            self.assertEqual(vector.shape[0], policy_num_dims(policy_spec))
            self.assertEqual(len(restored), len(params))
            for expected, actual in zip(params, restored):
                self.assertTrue(jnp.array_equal(expected, actual))
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_minatar_policy_spec_emits_conv_and_dense_shapes(self):
        try:
            import gymnax

            config = self._make_minatar_config()
            env, env_params = gymnax.make(config.env_id)
            obs_shape = tuple(int(dim) for dim in env.observation_space(env_params).shape)
            act_dim = int(env.action_space(env_params).n)

            policy_spec = policy_spec_for_task(config)
            conv_h = obs_shape[0] - 2
            conv_w = obs_shape[1] - 2
            flattened_dim = conv_h * conv_w * 16

            self.assertEqual(policy_spec.shapes[0], (3, 3, obs_shape[2], 16))
            self.assertEqual(policy_spec.shapes[1], (16,))
            self.assertEqual(policy_spec.shapes[2], (128, flattened_dim))
            self.assertEqual(policy_spec.shapes[3], (128,))
            self.assertEqual(policy_spec.shapes[4], (act_dim, 128))
            self.assertEqual(policy_spec.shapes[5], (act_dim,))
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_minatar_policy_vector_round_trip_with_conv_shapes(self):
        try:
            config = self._make_minatar_config()
            policy_spec = policy_spec_for_task(config)
            params = tuple(
                jnp.arange(size, dtype=jnp.float32).reshape(shape)
                for shape, size in zip(policy_spec.shapes, policy_spec.sizes)
            )
            vector = flatten_policy_params(params)
            restored = unflatten_policy_vector(vector, policy_spec)
            self.assertEqual(vector.shape[0], policy_num_dims(policy_spec))
            for expected, actual in zip(params, restored):
                self.assertTrue(jnp.array_equal(expected, actual))
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_minatar_discrete_reverse_rule(self):
        config = self._make_minatar_config(
            shift_windows=(ShiftWindowConfig(1, 3, "discrete_reverse"),),
        )
        reversed_action = map_action_for_shifts(
            jnp.asarray(0),
            jnp.asarray(2),
            config,
            act_dim=6,
            is_discrete=True,
        )
        self.assertEqual(int(reversed_action), 5)

    def test_apply_policy_mlp_path_matches_previous_behavior(self):
        config = make_config_gymnax_generic("CartPole-v1")
        obs = jnp.asarray([[1.0, -2.0]], dtype=jnp.float32)
        params = (
            jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float32),
            jnp.asarray([0.25, -0.75], dtype=jnp.float32),
            jnp.asarray([[1.5, -0.5]], dtype=jnp.float32),
            jnp.asarray([0.1], dtype=jnp.float32),
        )
        hidden = jnp.tanh(jnp.ravel(obs) @ params[0].T + params[1])
        expected = hidden @ params[2].T + params[3]
        actual = apply_policy(params, obs, config, is_discrete=False)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_apply_policy_cnn_mlp_output_shape(self):
        config = make_config_gymnax_generic(
            "Asterix-MinAtar",
            policy_architecture="cnn_mlp",
            policy_conv_channels=(16,),
            policy_conv_kernel_sizes=((3, 3),),
            policy_conv_strides=((1, 1),),
            policy_hidden_dims=(8,),
        )
        obs = jnp.ones((4, 4, 1), dtype=jnp.float32)
        params = (
            jnp.zeros((3, 3, 1, 16), dtype=jnp.float32),
            jnp.zeros((16,), dtype=jnp.float32),
            jnp.zeros((8, 64), dtype=jnp.float32),
            jnp.zeros((8,), dtype=jnp.float32),
            jnp.zeros((3, 8), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
        )
        output = apply_policy(params, obs, config, is_discrete=False)
        self.assertEqual(output.shape, (3,))

    def test_evosax_fitness_sign_conversion(self):
        raw_fitness = jnp.asarray([1.5, -2.0], dtype=jnp.float32)
        converted = fitness_for_evosax(raw_fitness)
        self.assertTrue(jnp.array_equal(converted, jnp.asarray([-1.5, 2.0], dtype=jnp.float32)))

    def test_evosax_strategy_resolution(self):
        try:
            available = available_evosax_algorithms()
            self.assertTrue(len(available) > 0)
            _ = resolve_evosax_strategy(available[0])
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        with self.assertRaises(ValueError):
            resolve_evosax_strategy("definitely_not_a_real_strategy")

    def test_population_based_evosax_adapter_init(self):
        try:
            if "samr_ga" not in available_evosax_algorithms():
                self.skipTest("samr_ga is not available in the installed evosax version.")
            from evosax_adapter import EvosaxStrategyAdapter

            config = make_config_nonstationary_gymnax(
                "CartPole-v1",
                optimizer_family="evosax",
                evosax_algo="samr_ga",
                pop_size=4,
                num_generations=2,
            )
            adapter = EvosaxStrategyAdapter(config, solution=jnp.zeros((3,), dtype=jnp.float32))
            self.assertTrue(adapter.requires_population_init)
            population = adapter.sample_initial_population(jax.random.key(0))
            self.assertEqual(population.shape, (config.pop_size, 3))
            state = adapter.init(
                jax.random.key(1),
                population,
                jnp.zeros((config.pop_size,), dtype=jnp.float32),
            )
            asked_population, ask_state = adapter.ask(jax.random.key(2), state)
            self.assertEqual(asked_population.shape, (config.pop_size, 3))
            next_state = adapter.tell(
                jax.random.key(3),
                asked_population,
                jnp.zeros((config.pop_size,), dtype=jnp.float32),
                ask_state,
            )
            self.assertIsNotNone(next_state)
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_evosax_wandb_payload_uses_effective_params(self):
        try:
            config = make_config_nonstationary_gymnax(
                "CartPole-v1",
                optimizer_family="evosax",
                evosax_algo="open_es",
            )
            policy_spec = policy_spec_for_task(config)
            payload = wandb_config_payload(config, policy_spec)
            self.assertNotIn("baseline_name", payload)
            self.assertIn("evosax_sigma_init_override", payload)
            self.assertIn("evosax_effective_params", payload)
            self.assertIn("policy_num_dims", payload)
            self.assertIsInstance(payload["evosax_effective_params"], dict)
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_mutation_metadata_is_finite(self):
        try:
            config = make_config_nonstationary_gymnax("CartPole-v1", pop_size=2, num_generations=2)
            graphs, specs = build_graphs_and_specs(config)
            pop = init_population(jax.random.key(0), config, graphs, specs)
            pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
            indiv = eqx.combine(jax.tree_util.tree_map(lambda x: x[0], pop_arr), pop_static)
            child, metadata = mutate_with_metadata(indiv, jax.random.key(1))
            self.assertTrue(jnp.isfinite(metadata.mutation_rate_mean))
            self.assertTrue(jnp.isfinite(metadata.update_rms))
            self.assertGreaterEqual(float(metadata.mutation_block_fraction), 0.0)
            self.assertIn("stoch", indiv.self_spec.module_names)
            self.assertFalse(jnp.allclose(indiv.stoch.basis, child.stoch.basis))
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_meta_sine_srghn_smoke_runs_end_to_end(self):
        cfg = self._make_meta_sine_config()
        cond = parse_condition_spec("srghn_self_self")
        payload = run_condition(cfg, cond)

        self.assertEqual(payload["kind"], "srghn")
        self.assertEqual(payload["train_history"]["fitness_best"].shape, (cfg.outer_generations,))
        self.assertEqual(payload["adaptation_curve_query_mse"]["mean"].shape, (cfg.inner_generations + 1,))
        self.assertEqual(payload["final_population_fitness"].shape, (cfg.outer_pop_size,))
        self.assertTrue(np.isfinite(payload["train_history"]["fitness_best"]).all())
        self.assertTrue(np.isfinite(payload["adaptation_curve_query_mse"]["mean"]).all())
        self.assertTrue(np.isfinite(payload["final_population_fitness"]).all())

    def test_meta_sine_srghn_is_deterministic(self):
        cfg = self._make_meta_sine_config()
        cond = parse_condition_spec("srghn_self_self")
        first = run_condition(cfg, cond)
        second = run_condition(cfg, cond)

        for key in first["train_history"]:
            np.testing.assert_allclose(first["train_history"][key], second["train_history"][key])
        self._assert_array_pytree_allclose(first["champion"], second["champion"])
        np.testing.assert_allclose(first["final_population_fitness"], second["final_population_fitness"])
        np.testing.assert_allclose(first["adaptation_curve_query_mse"]["mean"], second["adaptation_curve_query_mse"]["mean"])
        np.testing.assert_allclose(first["adaptation_curve_query_mse"]["stderr"], second["adaptation_curve_query_mse"]["stderr"])

    def test_generational_evo_step_selects_reproducers_and_evaluates_only_children(self):
        config = SimpleNamespace(
            pop_size=6,
            children_per_parent=2,
            srghn_replacement_mode="generational",
            mutation_exclude_modules=(),
            fixed_mutation_lr=None,
            num_generations=2,
            shift_windows=(),
        )
        state = EvoState(
            pop=jnp.arange(6, dtype=jnp.float32),
            key=jax.random.PRNGKey(0),
            obs_norm=jnp.asarray(0.0),
            pop_fitness=jnp.arange(6, dtype=jnp.float32),
        )
        evaluations = []

        def fake_eval(indiv, key, gen, cfg, obs_norm):
            del key, gen, cfg, obs_norm
            jax.debug.callback(lambda _: evaluations.append(1), indiv)
            return indiv, jnp.ones((1,)), jnp.ones((1,)), jnp.ones((1,))

        def fake_mutation(indiv, key, **kwargs):
            del key, kwargs
            return indiv + 100.0, jnp.asarray(0.0)

        with (
            mock.patch.object(evolution_module, "evaluate_individual_with_obs_stats", side_effect=fake_eval),
            mock.patch.object(evolution_module, "mutation_metadata", return_value=jnp.asarray(0.0)),
            mock.patch.object(evolution_module, "mutate_with_metadata", side_effect=fake_mutation),
            mock.patch.object(evolution_module, "compute_experiment_metrics", side_effect=lambda pop, fitness, parent, elite: {"fitness_mean": jnp.mean(fitness)}),
        ):
            next_state, _ = evolution_module.evo_step(state, jnp.asarray(1, dtype=jnp.int32), config)

        self.assertEqual(len(evaluations), 6)
        np.testing.assert_array_equal(np.asarray(next_state.pop), np.asarray([103, 103, 104, 104, 105, 105], dtype=np.float32))
        self.assertEqual(next_state.pop_fitness.shape, (6,))

    def test_native_elitist_evo_step_evaluates_25_parents_plus_175_children(self):
        config = SimpleNamespace(
            pop_size=25,
            children_per_parent=7,
            srghn_replacement_mode="elitist_union",
            mutation_exclude_modules=(),
            fixed_mutation_lr=None,
            child_factor=0.0,
            num_generations=1,
            shift_windows=(),
        )
        state = EvoState(
            pop=jnp.arange(25, dtype=jnp.float32),
            key=jax.random.PRNGKey(0),
            obs_norm=jnp.asarray(0.0),
            pop_fitness=jnp.zeros((25,), dtype=jnp.float32),
        )
        evaluations = []

        def fake_eval(indiv, key, gen, cfg, obs_norm):
            del key, gen, cfg, obs_norm
            jax.debug.callback(lambda _: evaluations.append(1), indiv)
            return indiv, jnp.ones((1,)), jnp.ones((1,)), jnp.ones((1,))

        def fake_mutation(indiv, key, **kwargs):
            del key, kwargs
            return indiv + 100.0, jnp.asarray(0.0)

        with (
            mock.patch.object(evolution_module, "evaluate_individual_with_obs_stats", side_effect=fake_eval),
            mock.patch.object(evolution_module, "mutation_metadata", return_value=jnp.asarray(0.0)),
            mock.patch.object(evolution_module, "mutate_with_metadata", side_effect=fake_mutation),
            mock.patch.object(evolution_module, "compute_experiment_metrics", side_effect=lambda pop, fitness, parent, elite: {"fitness_mean": jnp.mean(fitness)}),
            mock.patch.object(evolution_module, "update_obs_norm", side_effect=lambda current, *args: current),
        ):
            next_state, _ = evolution_module.evo_step(state, jnp.asarray(0, dtype=jnp.int32), config)

        self.assertEqual(len(evaluations), 200)
        self.assertEqual(next_state.pop.shape, (25,))
        self.assertEqual(next_state.pop_fitness.shape, (25,))

    def test_cached_elitist_evo_step_handles_stationary_and_switch_generations(self):
        def make_config(shift_windows):
            return SimpleNamespace(
                pop_size=6,
                children_per_parent=2,
                srghn_replacement_mode="cached_elitist",
                mutation_exclude_modules=(),
                fixed_mutation_lr=None,
                num_generations=2,
                shift_windows=shift_windows,
            )

        def fake_eval(indiv, key, gen, cfg, obs_norm):
            del key, gen, cfg, obs_norm
            jax.debug.callback(lambda _: evaluations.append(1), indiv)
            return jnp.where(indiv >= 100.0, indiv - 100.0, indiv), jnp.ones((1,)), jnp.ones((1,)), jnp.ones((1,))

        def fake_mutation(indiv, key, **kwargs):
            del key, kwargs
            return indiv + 100.0, jnp.asarray(0.0)

        with (
            mock.patch.object(evolution_module, "evaluate_individual_with_obs_stats", side_effect=fake_eval),
            mock.patch.object(evolution_module, "mutation_metadata", return_value=jnp.asarray(0.0)),
            mock.patch.object(evolution_module, "mutate_with_metadata", side_effect=fake_mutation),
            mock.patch.object(evolution_module, "compute_experiment_metrics", side_effect=lambda pop, fitness, parent, elite: {"fitness_mean": jnp.mean(fitness)}),
        ):
            evaluations = []
            stationary_state = EvoState(jnp.arange(6, dtype=jnp.float32), jax.random.PRNGKey(0), jnp.asarray(0.0), jnp.arange(6, dtype=jnp.float32))
            stationary_next, _ = evolution_module.evo_step(stationary_state, jnp.asarray(1, dtype=jnp.int32), make_config(()))
            self.assertEqual(len(evaluations), 6)
            stationary_pop = np.asarray(stationary_next.pop)
            stationary_fitness = np.asarray(stationary_next.pop_fitness)
            self.assertIn(4.0, stationary_pop)
            self.assertIn(5.0, stationary_pop)
            self.assertEqual(float(stationary_fitness[np.where(stationary_pop == 4.0)[0][0]]), 4.0)

            evaluations.clear()
            switch_state = EvoState(jnp.arange(6, dtype=jnp.float32), jax.random.PRNGKey(0), jnp.asarray(0.0), jnp.arange(6, dtype=jnp.float32))
            switch_next, _ = evolution_module.evo_step(
                switch_state,
                jnp.asarray(1, dtype=jnp.int32),
                make_config((ShiftWindowConfig(1, None, "discrete_reverse"),)),
            )
            self.assertEqual(len(evaluations), 6)
            np.testing.assert_array_equal(np.asarray(switch_next.pop), np.asarray([103, 103, 104, 104, 105, 105], dtype=np.float32))

    def test_short_cartpole_smoke_runs_all_baselines(self):
        try:
            for baseline in (
                BASELINE_FULL,
                BASELINE_FROZEN_MUTATION,
                BASELINE_FIXED_LR,
                BASELINE_NO_SELF_REFERENCE,
            ):
                config = make_config_nonstationary_gymnax(
                    "CartPole-v1",
                    baseline_name=baseline,
                    pop_size=2,
                    num_generations=2,
                    episode_horizon=8,
                    shift_windows=(ShiftWindowConfig(1, None, "cartpole_flip"),),
                )
                graphs, specs = build_graphs_and_specs(config)
                pop = init_population(jax.random.key(0), config, graphs, specs)
                state = EvoState(
                    pop=pop,
                    key=jax.random.key(1),
                    obs_norm=init_obs_norm(4),
                    pop_fitness=jnp.zeros((config.pop_size,), dtype=jnp.float32),
                )
                _next_state, metrics = evo_step(state, jnp.asarray(0, dtype=jnp.int32), config)
                self.assertIn("population_mutation_rate_mean", metrics)
                self.assertIn("elite_update_rms_mean", metrics)
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_short_evosax_cartpole_smoke_run(self):
        try:
            available = available_evosax_algorithms()
            algo = available[0]
            _ = resolve_evosax_strategy(algo)
            config = make_config_nonstationary_gymnax(
                "CartPole-v1",
                optimizer_family="evosax",
                baseline_name="evosax",
                evosax_algo=algo,
                pop_size=4,
                num_generations=2,
                episode_horizon=8,
                shift_windows=(ShiftWindowConfig(1, None, "cartpole_flip"),),
            )
            _state, metrics = run_experiment(config)
            self.assertIn("fitness_best", metrics)
            self.assertIn("population_mutation_rate_mean", metrics)
            self.assertIn("active_shift_windows", metrics)
            self.assertEqual(comparison_label(config), f"evosax:{algo}")
            self.assertEqual(metrics["fitness_best"].shape[0], config.num_generations)
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_short_minatar_smoke_run(self):
        try:
            config = self._make_minatar_config(
                pop_size=2,
                num_generations=1,
                episode_horizon=8,
                shift_windows=(ShiftWindowConfig(0, None, "discrete_reverse"),),
            )
            graphs, specs = build_graphs_and_specs(config)
            self.assertGreater(specs.policy_spec.num_nodes, 0)
            _state, metrics = run_experiment(config)
            self.assertIn("fitness_best", metrics)
            self.assertEqual(metrics["fitness_best"].shape[0], config.num_generations)
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_short_brax_smoke_run(self):
        try:
            config = make_config_nonstationary_brax(
                "ant",
                pop_size=2,
                num_generations=2,
                episode_horizon=8,
                shift_windows=(ShiftWindowConfig(1, None, "brax_direction_switch"),),
            )
            graphs, specs = build_graphs_and_specs(config)
            pop = init_population(jax.random.key(0), config, graphs, specs)
            _, _, obs_dim, _, _, _, _, _ = make_env(config)
            state = EvoState(
                pop=pop,
                key=jax.random.key(1),
                obs_norm=init_obs_norm(obs_dim),
                pop_fitness=jnp.zeros((config.pop_size,), dtype=jnp.float32),
            )
            _next_state, metrics = evo_step(state, jnp.asarray(0, dtype=jnp.int32), config)
            self.assertIn("fitness_best", metrics)
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_dict_based_observation_metadata_is_supported(self):
        from envs import _infer_obs_dim as infer_env_obs_dim

        class DummyEnv:
            observation_size = {"proprio": 8, "vision": {"left": 12, "right": 12}}
            obs_size = {"unused": 999}

        class DummySpace:
            def __init__(self, shape):
                self.shape = shape

        class DummySpaceEnv:
            def observation_space(self):
                return {"state": DummySpace((5,)), "extra": DummySpace((7,))}

        self.assertEqual(infer_env_obs_dim(DummyEnv()), 32)
        self.assertEqual(infer_specs_obs_dim(DummyEnv()), 32)
        self.assertEqual(infer_env_obs_dim(DummySpaceEnv()), 12)
        self.assertEqual(infer_specs_obs_dim(DummySpaceEnv()), 12)

    def test_dict_observations_flatten_for_normalization(self):
        obs = {"proprio": jnp.asarray([1.0, 2.0], dtype=jnp.float32), "vision": {"left": jnp.asarray([3.0])}}
        flat = flatten_observation(obs)
        self.assertEqual(tuple(flat.tolist()), (1.0, 2.0, 3.0))
        normed = normalize_obs(obs, init_obs_norm(3), clip=5.0, eps=1e-8)
        self.assertEqual(normed.shape, (3,))

    def test_cnn_observations_can_be_normalized_without_losing_spatial_shape(self):
        config = self._make_minatar_config()
        obs = jnp.ones((4, 4, 1), dtype=jnp.float32)
        state = init_obs_norm(16)
        normed = normalize_obs(obs, state, clip=5.0, eps=1e-8, preserve_shape=True)
        self.assertEqual(normed.shape, (4, 4, 1))

        params = (
            jnp.zeros((3, 3, 1, 16), dtype=jnp.float32),
            jnp.zeros((16,), dtype=jnp.float32),
            jnp.zeros((8, 64), dtype=jnp.float32),
            jnp.zeros((8,), dtype=jnp.float32),
            jnp.zeros((3, 8), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
        )
        output = apply_policy(params, normed, config, is_discrete=False)
        self.assertEqual(output.shape, (3,))


if __name__ == "__main__":
    unittest.main()
