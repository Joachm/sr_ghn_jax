from __future__ import annotations

from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from plot_ppo_experiments import (
    aggregate_ppo_metrics,
    discover_ppo_run_artifacts,
    main,
    plot_combined_overview,
    plot_per_environment,
)


def _write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _artifact(env_id: str, seed: int, *, optimizer_family: str = "ppo", baseline_name: str = "ppo", metrics: dict | None = None) -> dict:
    return {
        "config": SimpleNamespace(
            env_id=env_id,
            seed=seed,
            optimizer_family=optimizer_family,
            baseline_name=baseline_name,
        ),
        "metrics": metrics
        or {
            "fitness_best": np.asarray([1.0, 2.0, 3.0], dtype=float),
        },
    }


class PPOPlotTests(unittest.TestCase):
    def test_discover_ppo_run_artifacts_filters_to_ppo_and_env(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pickle(root / "ppo.pkl", _artifact("CartPole-v1", 0))
            _write_pickle(
                root / "srghn.pkl",
                _artifact(
                    "CartPole-v1",
                    1,
                    optimizer_family="srghn",
                    baseline_name="srghn_full",
                ),
            )

            records = discover_ppo_run_artifacts(root, env_ids=["CartPole-v1"])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["env_id"], "CartPole-v1")
            self.assertEqual(records[0]["seed"], 0)

    def test_aggregate_ppo_metrics_computes_mean_and_std(self):
        records = [
            {
                "path": Path("seed0.pkl"),
                "env_id": "CartPole-v1",
                "seed": 0,
                "fitness": np.asarray([1.0, 3.0, 5.0]),
            },
            {
                "path": Path("seed1.pkl"),
                "env_id": "CartPole-v1",
                "seed": 1,
                "fitness": np.asarray([3.0, 5.0, 7.0]),
            },
        ]

        aggregate = aggregate_ppo_metrics(records)
        entry = aggregate["CartPole-v1"]

        np.testing.assert_allclose(entry["fitness"]["mean"], np.asarray([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(entry["fitness"]["std"], np.asarray([1.0, 1.0, 1.0]))
        self.assertEqual(entry["num_runs"], 2)
        self.assertEqual(entry["seeds"], [0, 1])

    def test_aggregate_ppo_metrics_rejects_inconsistent_lengths(self):
        records = [
            {
                "path": Path("seed0.pkl"),
                "env_id": "CartPole-v1",
                "seed": 0,
                "fitness": np.asarray([1.0, 2.0, 3.0]),
            },
            {
                "path": Path("seed1.pkl"),
                "env_id": "CartPole-v1",
                "seed": 1,
                "fitness": np.asarray([1.0, 2.0]),
            },
        ]

        with self.assertRaisesRegex(ValueError, "Inconsistent update lengths"):
            aggregate_ppo_metrics(records)

    def test_plotting_smoke_writes_expected_files(self):
        aggregate = {
            "CartPole-v1": {
                "env_id": "CartPole-v1",
                "num_runs": 2,
                "seeds": [0, 1],
                "fitness": {
                    "mean": np.asarray([1.0, 2.0, 3.0]),
                    "std": np.asarray([0.1, 0.2, 0.3]),
                    "num_runs": 2,
                },
            },
            "Acrobot-v1": {
                "env_id": "Acrobot-v1",
                "num_runs": 1,
                "seeds": [0],
                "fitness": {
                    "mean": np.asarray([2.0, 3.0, 4.0]),
                    "std": np.asarray([0.0, 0.0, 0.0]),
                    "num_runs": 1,
                },
            },
        }
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            per_env_paths = plot_per_environment(aggregate, output_dir)
            combined_path = plot_combined_overview(aggregate, output_dir)

            self.assertEqual(len(per_env_paths), 2)
            self.assertTrue((output_dir / "cartpole_v1_fitness.png").exists())
            self.assertTrue((output_dir / "acrobot_v1_fitness.png").exists())
            self.assertEqual(combined_path, output_dir / "combined_fitness.png")
            self.assertTrue(combined_path.exists())

    def test_main_runs_end_to_end(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pickle(root / "run.pkl", _artifact("CartPole-v1", 0))
            output_dir = root / "plots"

            exit_code = main(["--results-root", str(root), "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "cartpole_v1_fitness.png").exists())
            self.assertTrue((output_dir / "combined_fitness.png").exists())


if __name__ == "__main__":
    unittest.main()
