from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import experiments.mujoco_playground_suite as suite


class MujocoPlaygroundSuiteTests(unittest.TestCase):
    def test_defaults_fan_out_ten_seeds(self):
        configs = []
        saved_paths = []
        metrics = {"fitness_best": [1, 2, 3], "fitness_mean": [1, 1, 1]}

        def fake_make_config(env_id, **kwargs):
            cfg = SimpleNamespace(
                env_backend="mujoco_playground",
                env_id=env_id,
                seed=kwargs["seed"],
                optimizer_family=kwargs.get("optimizer_family", "srghn"),
                baseline_name=kwargs.get("baseline_name", "srghn_full"),
                evosax_algo=kwargs.get("evosax_algo"),
                shift_windows=(),
                switch_gen_start=None,
                switch_gen_end=None,
                switch_rule=None,
                num_generations=kwargs.get("num_generations", 20000),
                pop_size=kwargs.get("pop_size", 200),
                children_per_parent=kwargs.get("children_per_parent", 8),
            )
            configs.append(cfg)
            return cfg

        def fake_run_experiment(config):
            return None, metrics

        def fake_save_pickle(path, payload):
            saved_paths.append(Path(path))

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = ["--env-id", "humanoid:run", "--output-dir", tmpdir]
            with mock.patch.object(suite, "make_config_mujoco_playground_generic", side_effect=fake_make_config), \
                mock.patch.object(suite, "run_experiment", side_effect=fake_run_experiment), \
                mock.patch.object(suite, "save_pickle", side_effect=fake_save_pickle), \
                mock.patch("sys.argv", ["prog", *argv]):
                suite.main()

        self.assertEqual([cfg.seed for cfg in configs], list(range(10)))
        self.assertEqual(len(saved_paths), 10)
        self.assertEqual(saved_paths[0], Path(tmpdir) / "srghn_full_g20000_p200_cpp8" / "humanoid_run" / "seed_0.pkl")
        self.assertEqual(saved_paths[-1], Path(tmpdir) / "srghn_full_g20000_p200_cpp8" / "humanoid_run" / "seed_9.pkl")


if __name__ == "__main__":
    unittest.main()
