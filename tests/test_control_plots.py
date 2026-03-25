from __future__ import annotations

from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from plot_control_experiments import (
    aggregate_control_metrics,
    build_control_colors,
    discover_run_artifacts,
    main,
    plot_combined_overview,
    plot_per_environment,
)


def _write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _artifact(env_id: str, seed: int, *, baseline_name: str = "srghn_full", optimizer_family: str = "srghn", evosax_algo: str | None = None, metrics: dict | None = None) -> dict:
    config_kwargs = {
        "env_id": env_id,
        "seed": seed,
        "optimizer_family": optimizer_family,
    }
    if optimizer_family == "evosax":
        config_kwargs["evosax_algo"] = evosax_algo
    else:
        config_kwargs["baseline_name"] = baseline_name

    return {
        "config": SimpleNamespace(**config_kwargs),
        "metrics": metrics
        or {
            "fitness_mean": np.asarray([1.0, 2.0, 3.0], dtype=float),
            "fitness_best": np.asarray([2.0, 3.0, 4.0], dtype=float),
            "diversity": np.asarray([0.5, 0.4, 0.3], dtype=float),
        },
    }


class ControlPlotTests(unittest.TestCase):
    def test_discover_run_artifacts_applies_filters_and_derives_control_labels(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pickle(root / "run_a.pkl", _artifact("CartPole-v1", 0, baseline_name="fixed_lr"))
            _write_pickle(
                root / "nested" / "run_b.pkl",
                _artifact("CartPole-v1", 1, optimizer_family="evosax", evosax_algo="OpenES"),
            )
            _write_pickle(root / "aggregate.pkl", {"aggregate": {"not": "a run artifact"}})

            records = discover_run_artifacts(root, env_ids=["CartPole-v1"], controls=["evosax:OpenES"])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["env_id"], "CartPole-v1")
            self.assertEqual(records[0]["control_label"], "evosax:OpenES")
            self.assertEqual(records[0]["seed"], 1)

    def test_aggregate_control_metrics_computes_mean_and_std(self):
        records = [
            {
                "path": Path("seed0.pkl"),
                "env_id": "CartPole-v1",
                "seed": 0,
                "control_label": "srghn_full",
                "metrics": {
                    "fitness_mean": np.asarray([1.0, 2.0, 3.0]),
                    "fitness_best": np.asarray([2.0, 4.0, 6.0]),
                    "diversity": np.asarray([0.2, 0.3, 0.4]),
                },
            },
            {
                "path": Path("seed1.pkl"),
                "env_id": "CartPole-v1",
                "seed": 1,
                "control_label": "srghn_full",
                "metrics": {
                    "fitness_mean": np.asarray([3.0, 4.0, 5.0]),
                    "fitness_best": np.asarray([4.0, 6.0, 8.0]),
                    "diversity": np.asarray([0.4, 0.5, 0.6]),
                },
            },
        ]

        aggregate = aggregate_control_metrics(records)
        entry = aggregate[("CartPole-v1", "srghn_full")]

        np.testing.assert_allclose(entry["metrics"]["fitness_mean"]["mean"], np.asarray([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(entry["metrics"]["fitness_mean"]["std"], np.asarray([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(entry["metrics"]["diversity"]["mean"], np.asarray([0.3, 0.4, 0.5]))
        self.assertEqual(entry["num_runs"], 2)
        self.assertEqual(entry["metrics"]["fitness_mean"]["num_runs"], 2)
        self.assertEqual(entry["seeds"], [0, 1])

    def test_aggregate_control_metrics_rejects_inconsistent_lengths(self):
        records = [
            {
                "path": Path("seed0.pkl"),
                "env_id": "CartPole-v1",
                "seed": 0,
                "control_label": "fixed_lr",
                "metrics": {
                    "fitness_mean": np.asarray([1.0, 2.0, 3.0]),
                    "fitness_best": np.asarray([1.0, 2.0, 3.0]),
                    "diversity": np.asarray([0.1, 0.2, 0.3]),
                },
            },
            {
                "path": Path("seed1.pkl"),
                "env_id": "CartPole-v1",
                "seed": 1,
                "control_label": "fixed_lr",
                "metrics": {
                    "fitness_mean": np.asarray([1.0, 2.0]),
                    "fitness_best": np.asarray([1.0, 2.0, 3.0]),
                    "diversity": np.asarray([0.1, 0.2, 0.3]),
                },
            },
        ]

        with self.assertRaisesRegex(ValueError, "Inconsistent generation lengths"):
            aggregate_control_metrics(records)

    def test_discover_run_artifacts_requires_all_target_metrics(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pickle(
                root / "broken.pkl",
                _artifact(
                    "CartPole-v1",
                    0,
                    metrics={
                        "fitness_mean": np.asarray([1.0, 2.0, 3.0]),
                        "fitness_best": np.asarray([2.0, 3.0, 4.0]),
                    },
                ),
            )

            with self.assertRaisesRegex(ValueError, "missing required metrics: diversity"):
                discover_run_artifacts(root)

    def test_plotting_smoke_writes_expected_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pickle(root / "cartpole_seed0.pkl", _artifact("CartPole-v1", 0, baseline_name="srghn_full"))
            _write_pickle(root / "cartpole_seed1.pkl", _artifact("CartPole-v1", 1, baseline_name="fixed_lr"))
            _write_pickle(root / "acrobot_seed0.pkl", _artifact("Acrobot-v1", 0, baseline_name="srghn_full"))
            _write_pickle(root / "acrobot_seed1.pkl", _artifact("Acrobot-v1", 1, baseline_name="fixed_lr"))

            records = discover_run_artifacts(root)
            aggregate = aggregate_control_metrics(records)
            colors = build_control_colors(aggregate)
            output_dir = root / "plots"

            per_env_paths = plot_per_environment(aggregate, output_dir, colors)
            combined_paths = plot_combined_overview(aggregate, output_dir, colors)

            self.assertEqual(len(per_env_paths), 6)
            self.assertEqual(len(combined_paths), 3)
            self.assertTrue((output_dir / "cartpole_v1_fitness_mean.png").exists())
            self.assertTrue((output_dir / "cartpole_v1_fitness_best.png").exists())
            self.assertTrue((output_dir / "cartpole_v1_diversity.png").exists())
            self.assertTrue((output_dir / "combined_fitness_mean.png").exists())
            self.assertTrue((output_dir / "combined_fitness_best.png").exists())
            self.assertTrue((output_dir / "combined_diversity.png").exists())

    def test_main_runs_end_to_end(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_pickle(root / "run.pkl", _artifact("CartPole-v1", 0))
            output_dir = root / "plots"

            exit_code = main(["--results-root", str(root), "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "cartpole_v1_fitness_mean.png").exists())


if __name__ == "__main__":
    unittest.main()
