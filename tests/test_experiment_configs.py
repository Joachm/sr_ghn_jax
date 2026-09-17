from __future__ import annotations

import importlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from adaptation_analysis import build_run_artifact
from configs import make_config_gymnax_generic, make_config_nonstationary_gymnax
from experiment_configs import (
    build_meta_brax_config,
    build_meta_sine_config,
    config_to_dict,
    resolve_gymnax_generic_config,
    resolve_nonstationary_brax_config,
    resolve_nonstationary_gymnax_config,
    resolve_ppo_nonstationary_gymnax_config,
)
from experiment_configs.catalog import render_experiment_catalog_markdown
from experiments._adaptation import gymnax_suite_shift_windows
from experiments.gymnax_minatar_suite import population_for_eval_budget


class ExperimentConfigTests(unittest.TestCase):
    def test_minatar_parser_preserves_legacy_replacement_default(self):
        module = importlib.import_module("experiments.gymnax_minatar_suite")
        args = module.parse_args([])
        self.assertEqual(args.srghn_replacement_mode, "elitist_union")
        self.assertEqual(args.children_per_parent, 8)

    def test_minatar_population_resolves_from_candidate_budget(self):
        self.assertEqual(population_for_eval_budget("evosax", 200, 1), 200)
        self.assertEqual(population_for_eval_budget("evosax", 200, 4), 200)
        self.assertEqual(population_for_eval_budget("srghn", 200, 4), 40)
        self.assertEqual(population_for_eval_budget("srghn", 200, 7, "elitist_union"), 25)
        self.assertEqual(population_for_eval_budget("srghn", 200, 2, "generational"), 200)
        self.assertEqual(population_for_eval_budget("srghn", 200, 2, "cached_elitist"), 200)
        with self.assertRaises(ValueError):
            population_for_eval_budget("srghn", 201, 2, "generational")
        with self.assertRaises(ValueError):
            population_for_eval_budget("srghn", 200, 8)

    def test_control_resolver_matches_stationary_gymnax_factory_defaults(self):
        expected = make_config_gymnax_generic("CartPole-v1")
        actual = resolve_gymnax_generic_config("CartPole-v1")
        self.assertEqual(config_to_dict(actual), config_to_dict(expected))

    def test_nonstationary_suite_task_preset_matches_existing_shift_setup(self):
        expected = make_config_nonstationary_gymnax("CartPole-v1", shift_windows=gymnax_suite_shift_windows("CartPole-v1"))
        actual = resolve_nonstationary_gymnax_config("CartPole-v1", task_preset="suite_default")
        self.assertEqual(config_to_dict(actual)["shift_windows"], config_to_dict(expected)["shift_windows"])

    def test_control_cli_override_beats_run_preset(self):
        config = resolve_nonstationary_brax_config(
            "ant",
            task_preset="brax_direction_switch",
            run_preset="debug",
            overrides={"num_generations": 9, "episode_horizon": 77},
        )
        self.assertEqual(config.num_generations, 9)
        self.assertEqual(config.episode_horizon, 77)

    def test_meta_sine_builder_uses_canonical_defaults(self):
        config = build_meta_sine_config()
        self.assertEqual(config.outer_generations, 1800)
        self.assertEqual(config.mutation_block_ratio, 1.0)
        self.assertEqual(config.wandb_project, "meta_sine_srghn_3")

    def test_meta_sine_cli_overrides_beat_fast_preset(self):
        module = importlib.import_module("meta_sine_srghn")
        args = module.parse_args(["--fast", "--outer-generations", "9", "--no-wandb"])
        config = module.make_base_cfg(args)
        self.assertEqual(config.outer_generations, 9)
        self.assertIsNone(config.wandb_project)

    def test_meta_brax_cli_overrides_beat_fast_preset(self):
        module = importlib.import_module("meta_brax_heading")
        args = module.parse_args(["--fast", "--outer-generations", "5", "--env-id", "ant"])
        config = module.make_base_cfg(args)
        self.assertEqual(config.outer_generations, 5)
        self.assertEqual(config.env_id, "ant")

    def test_ppo_resolver_supports_debug_preset(self):
        config = resolve_ppo_nonstationary_gymnax_config("CartPole-v1", run_preset="debug")
        self.assertEqual(config.episode_horizon, 32)
        self.assertEqual(config.num_envs, 4)

    def test_run_artifact_contains_resolved_config(self):
        config = make_config_gymnax_generic("CartPole-v1")
        artifact = build_run_artifact(config, {"fitness_best": [1.0], "fitness_mean": [1.0]})
        self.assertIn("resolved_config", artifact)
        self.assertEqual(artifact["resolved_config"]["config"]["env_id"], "CartPole-v1")

    def test_print_config_available_across_training_entrypoints(self):
        cases = [
            ("experiments.ant_brax", []),
            ("experiments.brax_generic", ["--env-id", "ant"]),
            ("experiments.gymnax_generic", ["--env-id", "CartPole-v1"]),
            ("experiments.nonstationary_gymnax", ["--env-id", "CartPole-v1"]),
            ("experiments.nonstationary_brax", []),
            ("experiments.mujoco_playground_generic", ["--env-id", "humanoid:run"]),
            ("experiments.mujoco_playground_suite", ["--env-id", "humanoid:run"]),
            ("experiments.gymnax_nonstationary_suite", []),
            ("experiments.gymnax_minatar_suite", []),
            ("baselines.ppo.suite", []),
            ("meta_sine_srghn", []),
            ("meta_brax_heading", []),
        ]
        for module_name, extra_args in cases:
            module = importlib.import_module(module_name)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = module.main([*extra_args, "--print-config"])
            self.assertEqual(rc, 0, msg=module_name)
            payload = json.loads(stdout.getvalue())
            self.assertIn("config", payload)

    def test_minatar_print_config_keeps_derived_counts_in_json(self):
        module = importlib.import_module("experiments.gymnax_minatar_suite")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = module.main([
                "--print-config",
                "--optimizer-family",
                "srghn",
                "--eval-budget-per-generation",
                "200",
                "--children-per-parent",
                "2",
                "--srghn-replacement-mode",
                "cached_elitist",
            ])
        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        config = payload["config"]
        self.assertEqual(config["resident_population_size"], 200)
        self.assertEqual(config["num_reproducers"], 100)
        self.assertEqual(config["evaluated_candidates_per_generation"], 200)

    def test_minatar_evosax_print_config_does_not_require_srghn_fields(self):
        module = importlib.import_module("experiments.gymnax_minatar_suite")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = module.main([
                "--print-config",
                "--optimizer-family",
                "evosax",
                "--evosax-algo",
                "cma_es",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout.getvalue())["config"]["optimizer_family"], "evosax")

    def test_catalog_markdown_matches_checked_in_reference(self):
        doc_path = Path("docs/experiment_configuration_reference.md")
        self.assertEqual(doc_path.read_text(encoding="utf-8").strip(), render_experiment_catalog_markdown().strip())


if __name__ == "__main__":
    unittest.main()
