from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from adaptation_analysis import build_run_artifact, save_pickle
from baselines.ppo.config import PPO_GYMNAX_SUITE_ENVIRONMENTS, make_ppo_config_nonstationary_gymnax
from baselines.ppo.networks import init_actor_critic_params, policy_output_dim
from baselines.ppo.runner import _policy_action_to_env_action, _policy_obs_single, run_ppo
from baselines.ppo.suite import main as ppo_suite_main
from configs import ShiftWindowConfig
from experiments._adaptation import gymnax_suite_shift_windows
from experiments.gymnax_nonstationary_suite import ENVIRONMENTS as SRGHN_GYMNAX_SUITE_ENVIRONMENTS
from obs_norm import init_obs_norm
from plot_control_experiments import discover_run_artifacts
from specs import policy_spec_for_task


REQUIRED_COMPAT_METRICS = (
    "fitness_mean",
    "fitness_best",
    "fitness_min",
    "fitness_std",
    "fitness_median",
    "diversity",
    "population_mutation_rate_mean",
    "population_mutation_rate_std",
    "population_mutation_rate_max",
    "population_mutation_block_fraction_mean",
    "population_mutation_blocks_selected_mean",
    "population_mutation_total_blocks_mean",
    "population_update_rms_mean",
    "population_self_distance_rms_mean",
    "elite_mutation_rate_mean",
    "elite_mutation_rate_std",
    "elite_mutation_rate_max",
    "elite_mutation_block_fraction_mean",
    "elite_mutation_blocks_selected_mean",
    "elite_mutation_total_blocks_mean",
    "elite_update_rms_mean",
    "elite_self_distance_rms_mean",
    "active_shift_windows",
)


class PPOBaselineTests(unittest.TestCase):
    def _require_gymnax(self):
        try:
            import gymnax  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

    def test_suite_envs_match_existing_nonstationary_suite(self):
        self.assertEqual(PPO_GYMNAX_SUITE_ENVIRONMENTS, SRGHN_GYMNAX_SUITE_ENVIRONMENTS)

    def test_default_config_uses_suite_shift_windows(self):
        config = make_ppo_config_nonstationary_gymnax("CartPole-v1", seed=7)
        self.assertEqual(config.optimizer_family, "ppo")
        self.assertEqual(config.baseline_name, "ppo")
        self.assertEqual(config.shift_windows, gymnax_suite_shift_windows("CartPole-v1"))

    def test_policy_shift_helpers_match_nonstationary_rules(self):
        action_config = make_ppo_config_nonstationary_gymnax(
            "CartPole-v1",
            shift_windows=(ShiftWindowConfig(1, 3, "cartpole_flip"),),
        )
        flipped_action = _policy_action_to_env_action(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
            action_config,
            act_dim=2,
            is_discrete=True,
            action_shape=(),
            action_low=None,
            action_high=None,
        )
        self.assertEqual(int(flipped_action), 1)

        obs_config = make_ppo_config_nonstationary_gymnax(
            "Pendulum-v1",
            shift_windows=(ShiftWindowConfig(1, 3, "pendulum_obs_flip"),),
        )
        shifted_obs = _policy_obs_single(
            jnp.asarray([0.8, 0.2, 1.5], dtype=jnp.float32),
            jnp.asarray(2, dtype=jnp.int32),
            obs_config,
            init_obs_norm(3),
        )
        self.assertAlmostEqual(float(shifted_obs[0]), 0.2, places=5)
        self.assertAlmostEqual(float(shifted_obs[1]), 0.8, places=5)
        self.assertAlmostEqual(float(shifted_obs[2]), -1.5, places=5)

    def test_actor_output_shape_matches_policy_spec(self):
        self._require_gymnax()
        config = make_ppo_config_nonstationary_gymnax("MountainCarContinuous-v0")
        policy_spec = policy_spec_for_task(config)
        actor_output = policy_spec.shapes[-1][0]
        obs_dim = int(policy_spec.shapes[0][1])
        params = init_actor_critic_params(
            jax.random.key(0),
            obs_dim,
            actor_output,
            config.policy_hidden_dims,
            is_discrete=False,
        )
        self.assertEqual(policy_output_dim(params), actor_output)
        self.assertEqual(len(params["trunk"]), len(config.policy_hidden_dims))

    def test_short_run_returns_compatible_metrics(self):
        self._require_gymnax()
        config = make_ppo_config_nonstationary_gymnax(
            "CartPole-v1",
            seed=0,
            num_generations=2,
            episode_horizon=8,
            num_envs=2,
            rollout_length=4,
            num_minibatches=2,
            update_epochs=1,
            shift_windows=(ShiftWindowConfig(1, None, "cartpole_flip"),),
        )
        _state, metrics = run_ppo(config)
        for metric_name in REQUIRED_COMPAT_METRICS:
            self.assertIn(metric_name, metrics)
            self.assertEqual(metrics[metric_name].shape[0], config.num_generations)
        self.assertEqual(metrics["actor_loss"].shape[0], config.num_generations)

    def test_artifact_is_discoverable_by_control_plotting(self):
        self._require_gymnax()
        config = make_ppo_config_nonstationary_gymnax(
            "CartPole-v1",
            seed=1,
            num_generations=1,
            episode_horizon=8,
            num_envs=2,
            rollout_length=4,
            num_minibatches=2,
            update_epochs=1,
        )
        _state, metrics = run_ppo(config)
        artifact = build_run_artifact(config, metrics)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ppo_run.pkl"
            save_pickle(path, artifact)
            records = discover_run_artifacts(path.parent)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["control_label"], "ppo")
        self.assertEqual(records[0]["env_id"], "CartPole-v1")

    def test_suite_smoke_writes_result_artifact(self):
        self._require_gymnax()
        with TemporaryDirectory() as tmpdir, mock.patch(
            "baselines.ppo.suite.PPO_GYMNAX_SUITE_ENVIRONMENTS",
            ("CartPole-v1",),
        ):
            exit_code = ppo_suite_main(
                [
                    "--seeds",
                    "0",
                    "--output-dir",
                    tmpdir,
                    "--num-generations",
                    "1",
                    "--episode-horizon",
                    "8",
                    "--num-envs",
                    "2",
                    "--rollout-length",
                    "4",
                    "--num-minibatches",
                    "2",
                    "--update-epochs",
                    "1",
                ]
            )
            self.assertEqual(exit_code, 0)
            artifacts = sorted(Path(tmpdir).rglob("seed_0.pkl"))
            self.assertEqual(len(artifacts), 1)


if __name__ == "__main__":
    unittest.main()
