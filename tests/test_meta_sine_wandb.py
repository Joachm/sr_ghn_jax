from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from types import ModuleType
import unittest

import numpy as np

import meta_sine_srghn as ms


class _FakeWandb:
    def __init__(self):
        self.module = ModuleType("wandb")
        self.module.run = None
        self.module.init_calls = []
        self.module.log_calls = []
        self.module.finish_calls = 0
        self.module.init = self.init
        self.module.log = self.log
        self.module.finish = self.finish

    def init(self, **kwargs):
        self.module.init_calls.append(kwargs)
        self.module.run = SimpleNamespace(name=kwargs.get("name"))
        return self.module.run

    def log(self, payload, step=None):
        self.module.log_calls.append({"payload": dict(payload), "step": step})

    def finish(self):
        self.module.finish_calls += 1
        self.module.run = None


def _tiny_cfg(**overrides) -> ms.MetaSineConfig:
    cfg = ms.MetaSineConfig(
        seed=0,
        outer_generations=2,
        meta_batch_size=2,
        test_task_batch_size=2,
        outer_pop_size=2,
        outer_children_per_parent=1,
        inner_pop_size=2,
        inner_children_per_parent=1,
        inner_generations=1,
        support_k=3,
        query_k=4,
        policy_hidden_dims=(4,),
        embedding_dim=4,
        gnn_hidden_dim=4,
        gnn_steps_policy=1,
        gnn_steps_self=1,
        stoch_coeff_dim=4,
        parameter_block_size=8,
        mutation_rate_head_dim=2,
        vector_ga_sigma=0.01,
        vector_ga_init_scale=0.01,
        wandb_project="meta_sine_srghn_test",
        wandb_group="wandb-smoke",
        wandb_name="unit",
        wandb_log_plots=False,
    )
    return replace(cfg, **overrides)


class MetaSineWandbTests(unittest.TestCase):
    def setUp(self):
        self.fake_wandb = _FakeWandb()
        self._old_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = self.fake_wandb.module

    def tearDown(self):
        if self._old_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = self._old_wandb

    def test_srghn_condition_logs_progress_and_wandb_config(self):
        cfg = _tiny_cfg()
        cond = ms.validate_condition(ms.PRESET_CONDITIONS["srghn_self_self"])

        result = ms.run_condition(cfg, cond)

        self.assertEqual(len(self.fake_wandb.module.init_calls), 1)
        self.assertEqual(self.fake_wandb.module.finish_calls, 1)
        self.assertEqual(sum("gen" in entry["payload"] for entry in self.fake_wandb.module.log_calls), cfg.outer_generations)
        self.assertEqual(self.fake_wandb.module.init_calls[0]["name"], "unit-srghn_self_self")
        self.assertEqual(self.fake_wandb.module.init_calls[0]["group"], "wandb-smoke")

        config_payload = self.fake_wandb.module.init_calls[0]["config"]
        self.assertEqual(config_payload["condition"]["name"], "srghn_self_self")
        self.assertIn("policy_spec_shapes", config_payload)
        self.assertIn("self_spec_shapes", config_payload)

        gen_steps = [entry["step"] for entry in self.fake_wandb.module.log_calls if "gen" in entry["payload"]]
        self.assertEqual(gen_steps, [0, 1])
        self.assertIn("train_history", result)
        self.assertIn("adaptation_curve_query_mse", result)
        self.assertEqual(result["config"]["wandb_project"], "meta_sine_srghn_test")

    def test_vector_condition_uses_same_logging_lifecycle_and_name_scheme(self):
        cfg = _tiny_cfg()
        cond = ms.validate_condition(ms.PRESET_CONDITIONS["vector_ga_ga"])

        result = ms.run_condition(cfg, cond)

        self.assertEqual(len(self.fake_wandb.module.init_calls), 1)
        self.assertEqual(self.fake_wandb.module.finish_calls, 1)
        self.assertEqual(self.fake_wandb.module.init_calls[0]["name"], "unit-vector_ga_ga")
        self.assertEqual(sum("gen" in entry["payload"] for entry in self.fake_wandb.module.log_calls), cfg.outer_generations)
        self.assertEqual(self.fake_wandb.module.init_calls[0]["config"]["condition"]["name"], "vector_ga_ga")
        self.assertIn("policy_num_dims", self.fake_wandb.module.init_calls[0]["config"])
        self.assertIn("final_population_fitness", result)

    def test_run_condition_is_deterministic_with_wandb_logging(self):
        cfg = _tiny_cfg()
        cond = ms.validate_condition(ms.PRESET_CONDITIONS["srghn_self_self"])

        first = ms.run_condition(cfg, cond)
        self.fake_wandb.module.init_calls.clear()
        self.fake_wandb.module.log_calls.clear()
        self.fake_wandb.module.finish_calls = 0
        self.fake_wandb.module.run = None
        second = ms.run_condition(cfg, cond)

        np.testing.assert_allclose(first["train_history"]["fitness_best"], second["train_history"]["fitness_best"])
        np.testing.assert_allclose(first["train_history"]["query_mse_best"], second["train_history"]["query_mse_best"])
        np.testing.assert_allclose(first["final_population_fitness"], second["final_population_fitness"])
        np.testing.assert_allclose(first["adaptation_curve_query_mse"]["mean"], second["adaptation_curve_query_mse"]["mean"])

    def test_none_project_disables_wandb_cleanly(self):
        cfg = _tiny_cfg(wandb_project=None)
        cond = ms.validate_condition(ms.PRESET_CONDITIONS["srghn_self_self"])

        ms.run_condition(cfg, cond)

        self.assertEqual(self.fake_wandb.module.init_calls, [])
        self.assertEqual(self.fake_wandb.module.log_calls, [])
        self.assertEqual(self.fake_wandb.module.finish_calls, 0)


if __name__ == "__main__":
    unittest.main()
