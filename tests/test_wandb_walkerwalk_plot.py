from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest import mock
from unittest.mock import patch
import sys

import numpy as np

from plot_wandb_walkerwalk import (
    aggregate_project_series,
    fetch_project_series,
    plot_project_series,
)


class _FakeRun:
    def __init__(self, run_id: str, name: str, rows: list[dict]):
        self.id = run_id
        self.name = name
        self._rows = rows

    def scan_history(self, keys=None):
        yield from self._rows


class _FakeApi:
    def __init__(self, runs: list[_FakeRun]):
        self._runs = runs

    def runs(self, project_path, order=None):
        self.project_path = project_path
        self.order = order
        return self._runs


class _FakeCmap:
    def __call__(self, idx):
        return (0.2, 0.4, 0.6, 1.0)


class _FakeFigure:
    def __init__(self):
        self.saved_paths = []

    def tight_layout(self):
        return None

    def savefig(self, path, dpi=None, bbox_inches=None):
        Path(path).write_bytes(b"fake image")
        self.saved_paths.append(Path(path))


class _FakeAxes:
    def plot(self, *args, **kwargs):
        return None

    def fill_between(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None


class _FakePyplot(ModuleType):
    def __init__(self):
        super().__init__("matplotlib.pyplot")

    def get_cmap(self, name):
        return _FakeCmap()

    def subplots(self, figsize=None):
        return _FakeFigure(), _FakeAxes()

    def close(self, fig):
        return None


class WandBWalkerWalkPlotTests(unittest.TestCase):
    def test_aggregate_project_series_aligns_runs_and_computes_mean_std(self):
        series_list = [
            {
                "run_id": "a",
                "run_name": "run-a",
                "steps": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "values": np.asarray([1.0, 3.0, 5.0], dtype=float),
            },
            {
                "run_id": "b",
                "run_name": "run-b",
                "steps": np.asarray([0.0, 2.0], dtype=float),
                "values": np.asarray([2.0, 6.0], dtype=float),
            },
        ]

        aggregate = aggregate_project_series(series_list)

        np.testing.assert_allclose(aggregate["steps"], np.asarray([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(aggregate["mean"], np.asarray([1.5, 3.5, 5.5]))
        np.testing.assert_allclose(aggregate["std"], np.asarray([0.5, 0.5, 0.5]))
        self.assertEqual(aggregate["num_runs"], 2)

    def test_fetch_project_series_uses_scan_history_rows(self):
        fake_api = _FakeApi(
            runs=[
                _FakeRun(
                    "run-1",
                    "first",
                    [
                        {"_step": 0, "fitness_best": 1.0},
                        {"_step": 1, "fitness_best": 2.0},
                    ],
                ),
                _FakeRun(
                    "run-2",
                    "second",
                    [
                        {"_step": 0, "fitness_best": 3.0},
                        {"_step": 2, "fitness_best": 5.0},
                    ],
                ),
            ]
        )

        with patch("plot_wandb_walkerwalk._load_wandb_api", return_value=fake_api):
            series_list = fetch_project_series("entity/WalkerWalk")

        self.assertEqual(fake_api.project_path, "entity/WalkerWalk")
        self.assertEqual(fake_api.order, "+created_at")
        self.assertEqual(len(series_list), 2)
        np.testing.assert_allclose(series_list[0]["steps"], np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(series_list[1]["values"], np.asarray([3.0, 5.0]))

    def test_fetch_project_series_supports_custom_metric_names(self):
        fake_api = _FakeApi(
            runs=[
                _FakeRun(
                    "run-1",
                    "first",
                    [
                        {"_step": 0, "score": 10.0},
                        {"_step": 1, "score": 11.0},
                    ],
                ),
            ]
        )

        with patch("plot_wandb_walkerwalk._load_wandb_api", return_value=fake_api):
            series_list = fetch_project_series("entity/OtherProject", metric_name="score")

        self.assertEqual(len(series_list), 1)
        np.testing.assert_allclose(series_list[0]["values"], np.asarray([10.0, 11.0]))

    def test_plot_project_series_writes_png(self):
        aggregate = {
            "steps": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "mean": np.asarray([1.0, 2.0, 3.0], dtype=float),
            "std": np.asarray([0.1, 0.2, 0.3], dtype=float),
            "num_runs": 2,
        }

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "walkerwalk.png"
            fake_matplotlib = ModuleType("matplotlib")
            fake_matplotlib.use = lambda *args, **kwargs: None
            fake_matplotlib.rcParams = {}
            fake_pyplot = _FakePyplot()
            fake_matplotlib.pyplot = fake_pyplot
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
            ):
                written = plot_project_series(
                    aggregate,
                    project_path="entity/WalkerWalk",
                    metric_name="fitness_best",
                    output_path=output_path,
                )

            self.assertEqual(written, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
