from __future__ import annotations

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

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
from experiments._adaptation import gymnax_suite_shift_windows
from experiments._common import build_graphs_and_specs, default_wandb_project, run_experiment
from evolution import evo_step, init_population, EvoState
from obs_norm import init_obs_norm
from policy_vectors import flatten_policy_params, policy_num_dims, unflatten_policy_vector
from srghn import mutate_with_metadata
from specs import policy_spec_for_task


class AdaptationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
