from __future__ import annotations

import importlib
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import configs
import meta_brax_heading as mb
from configs import BASELINE_FIXED_LR, BASELINE_FROZEN_MUTATION, BASELINE_FULL, BASELINE_NO_SELF_REFERENCE


def _make_pipeline_state(x: float, y: float):
    pos = jnp.asarray([[x, y, 0.0]], dtype=jnp.float32)
    return SimpleNamespace(x=SimpleNamespace(pos=pos))


def _make_state(*, metrics=None, x: float | None = None, y: float | None = None):
    kwargs = {}
    if metrics is not None:
        kwargs["metrics"] = metrics
    if x is not None and y is not None:
        kwargs["pipeline_state"] = _make_pipeline_state(x, y)
    return SimpleNamespace(**kwargs)


class MetaBraxHeadingTests(unittest.TestCase):
    def test_brax_config_helpers_are_exported(self):
        self.assertTrue(hasattr(configs, "make_config_ant_brax"))
        self.assertTrue(hasattr(configs, "make_config_brax_generic"))
        self.assertTrue(hasattr(configs, "make_config_nonstationary_brax"))
        self.assertTrue(callable(configs.make_config_nonstationary_brax))

    def test_nonstationary_brax_module_imports(self):
        module = importlib.import_module("experiments.nonstationary_brax")
        self.assertTrue(hasattr(module, "main"))

    def test_random_cardinal_heading_sampling_is_deterministic_and_cardinal(self):
        key = jax.random.PRNGKey(0)
        first = mb.sample_heading_tasks(key, 32)
        second = mb.sample_heading_tasks(key, 32)
        np.testing.assert_array_equal(np.asarray(first.headings), np.asarray(second.headings))

        allowed = {tuple(row.tolist()) for row in np.asarray(mb.CARDINAL_HEADINGS)}
        observed = {tuple(row.tolist()) for row in np.asarray(first.headings)}
        self.assertTrue(observed.issubset(allowed))

    def test_random_cardinal_heading_sampling_is_approximately_uniform(self):
        batch = mb.sample_heading_tasks(jax.random.PRNGKey(1), 4000)
        headings = np.asarray(batch.headings)
        allowed = np.asarray(mb.CARDINAL_HEADINGS)
        counts = []
        for heading in allowed:
            counts.append(int(np.all(headings == heading, axis=1).sum()))
        expected = headings.shape[0] / allowed.shape[0]
        for count in counts:
            self.assertLess(abs(count - expected), expected * 0.2)

    def test_fixed_training_heading_tasks_are_exactly_cardinal_set(self):
        batch = mb.fixed_training_heading_tasks(4)
        np.testing.assert_allclose(np.asarray(batch.headings), np.asarray(mb.CARDINAL_HEADINGS))

    def test_fixed_training_heading_tasks_require_four_tasks(self):
        with self.assertRaises(ValueError):
            mb.fixed_training_heading_tasks(3)

    def test_heldout_heading_sampling_is_deterministic_and_unit_norm(self):
        key = jax.random.PRNGKey(7)
        first = mb.sample_heldout_heading_tasks(key, 16)
        second = mb.sample_heldout_heading_tasks(key, 16)
        np.testing.assert_allclose(np.asarray(first.headings), np.asarray(second.headings))
        norms = np.linalg.norm(np.asarray(first.headings), axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_heldout_heading_sampling_is_not_restricted_to_cardinals(self):
        batch = mb.sample_heldout_heading_tasks(jax.random.PRNGKey(11), 64)
        headings = np.asarray(batch.headings)
        allowed = {tuple(row.tolist()) for row in np.asarray(mb.CARDINAL_HEADINGS)}
        observed = {tuple(np.round(row, 5).tolist()) for row in headings}
        self.assertFalse(observed.issubset(allowed))

    def test_support_and_query_episode_keys_are_disjoint(self):
        support, query = mb.split_support_query_episode_keys(jax.random.PRNGKey(2), 3, 4)
        support_data = np.asarray(jax.random.key_data(support))
        query_data = np.asarray(jax.random.key_data(query))
        for support_key in support_data:
            for query_key in query_data:
                self.assertFalse(np.array_equal(support_key, query_key))

    def test_projected_heading_reward_uses_metric_velocity_when_available(self):
        prev_state = _make_state(x=0.0, y=0.0)
        next_state = _make_state(metrics={"x_velocity": jnp.asarray(2.0), "y_velocity": jnp.asarray(-3.0)})
        headings = np.asarray(mb.CARDINAL_HEADINGS)
        expected = np.asarray([2.0, -2.0, -3.0, 3.0], dtype=np.float32)
        actual = np.asarray(
            [mb._projected_heading_reward(prev_state, next_state, jnp.asarray(heading), 0.1) for heading in headings]
        )
        np.testing.assert_allclose(actual, expected)

    def test_position_delta_fallback_matches_metric_velocity(self):
        prev_state = _make_state(x=0.0, y=0.0)
        next_state_metrics = _make_state(metrics={"x_velocity": jnp.asarray(2.0), "y_velocity": jnp.asarray(-3.0)})
        next_state_positions = _make_state(x=0.2, y=-0.3)
        dt = 0.1
        metric_velocity = mb._extract_planar_velocity(next_state_metrics, prev_state, dt)
        position_velocity = mb._extract_planar_velocity(next_state_positions, prev_state, dt)
        np.testing.assert_allclose(np.asarray(metric_velocity), np.asarray(position_velocity))

    def test_heading_label_and_choice_mapping(self):
        self.assertEqual(mb._heading_label(jnp.asarray([1.0, 0.0], dtype=jnp.float32)), "+x")
        self.assertEqual(mb._heading_label(jnp.asarray([0.0, -1.0], dtype=jnp.float32)), "-y")
        np.testing.assert_allclose(
            np.asarray(mb._showcase_heading_from_choice("pos_y")),
            np.asarray([0.0, 1.0], dtype=np.float32),
        )

    def test_downsample_frames_caps_frame_count(self):
        frames = np.zeros((100, 8, 8, 3), dtype=np.uint8)
        sampled = mb._downsample_frames(frames, 12)
        self.assertEqual(sampled.shape[0], 12)
        self.assertEqual(sampled.shape[1:], frames.shape[1:])

    def test_rollout_policy_params_on_heading_keeps_done_outside_jit(self):
        PosState = namedtuple("PosState", ["pos"])
        PipelineState = namedtuple("PipelineState", ["x"])
        RolloutState = namedtuple("RolloutState", ["obs", "done", "pipeline_state"])

        def make_pipeline_state_array(x_value):
            pos = jnp.stack(
                [
                    jnp.stack(
                        [
                            jnp.asarray(x_value, dtype=jnp.float32),
                            jnp.asarray(0.0, dtype=jnp.float32),
                            jnp.asarray(0.0, dtype=jnp.float32),
                        ]
                    )
                ]
            )
            return PipelineState(x=PosState(pos=pos))

        class DummyEnv:
            observation_size = 1
            action_size = 1

            def reset(self, key):
                del key
                return RolloutState(
                    obs=jnp.asarray([0.0], dtype=jnp.float32),
                    done=jnp.asarray(False),
                    pipeline_state=make_pipeline_state_array(0.0),
                )

            def step(self, state, action):
                del action
                current_x = state.pipeline_state.x.pos[0, 0]
                next_x = current_x + 0.1
                return RolloutState(
                    obs=jnp.asarray([next_x], dtype=jnp.float32),
                    done=jnp.asarray(next_x >= 0.2),
                    pipeline_state=make_pipeline_state_array(next_x),
                )

        cfg = mb.MetaBraxConfig(episode_horizon=4, query_episodes=1, support_episodes=1, wandb_project=None)
        heading = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
        episode_key = jax.random.PRNGKey(0)

        with (
            mock.patch.object(mb, "make_runtime_config", return_value=SimpleNamespace(env_backend="brax", obs_norm_clip=5.0, obs_norm_eps=1e-8)),
            mock.patch.object(mb, "make_env", return_value=(DummyEnv(), None, 1, 1, False, (1,), None, None)),
            mock.patch.object(mb, "_infer_env_dt", return_value=0.1),
            mock.patch.object(mb, "normalize_obs", side_effect=lambda obs, *_args, **_kwargs: obs),
            mock.patch.object(mb, "apply_policy", return_value=jnp.asarray([0.0], dtype=jnp.float32)),
        ):
            trajectory, total_reward = mb.rollout_policy_params_on_heading(object(), episode_key, heading, cfg)

        self.assertGreaterEqual(len(trajectory), 2)
        self.assertTrue(np.isfinite(total_reward))

    def test_baseline_condition_mapping_matches_existing_overrides(self):
        full = mb.parse_condition_spec(BASELINE_FULL, fixed_mutation_lr=0.02)
        frozen = mb.parse_condition_spec(BASELINE_FROZEN_MUTATION, fixed_mutation_lr=0.02)
        fixed = mb.parse_condition_spec(BASELINE_FIXED_LR, fixed_mutation_lr=0.02)
        no_self = mb.parse_condition_spec(BASELINE_NO_SELF_REFERENCE, fixed_mutation_lr=0.02)

        self.assertEqual(full.mutation_exclude_modules, ())
        self.assertIsNone(full.fixed_mutation_lr)

        self.assertEqual(frozen.mutation_exclude_modules, ("stoch",))
        self.assertIsNone(frozen.fixed_mutation_lr)

        self.assertEqual(fixed.mutation_exclude_modules, ())
        self.assertEqual(fixed.fixed_mutation_lr, 0.02)

        self.assertEqual(
            no_self.mutation_exclude_modules,
            ("self_node_emb", "self_context_emb", "self_feat_proj", "encoder_self", "stoch"),
        )
        self.assertIsNone(no_self.fixed_mutation_lr)

    def test_short_meta_brax_smoke_run(self):
        try:
            cfg = mb.MetaBraxConfig(
                env_id="ant",
                brax_backend="spring",
                seed=0,
                outer_generations=1,
                meta_batch_size=1,
                heldout_task_batch_size=1,
                outer_pop_size=2,
                outer_children_per_parent=1,
                inner_pop_size=2,
                inner_children_per_parent=1,
                inner_generations=1,
                support_episodes=1,
                query_episodes=1,
                episode_horizon=8,
                wandb_project=None,
            )
            cond = mb.parse_condition_spec(BASELINE_FULL, fixed_mutation_lr=cfg.baseline_fixed_mutation_lr)
            payload = mb.run_condition(cfg, cond)
            curve = payload["adaptation_curve_query_return"]["mean"]
            self.assertEqual(curve.shape, (cfg.inner_generations + 1,))
            self.assertTrue(np.isfinite(curve).all())
            self.assertTrue(np.isfinite(payload["train_history"]["query_return_best"]).all())
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))


if __name__ == "__main__":
    unittest.main()
