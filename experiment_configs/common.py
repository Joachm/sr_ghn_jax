from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def resolved_config_payload(
    config: Any,
    *,
    family: str | None = None,
    task_preset: str | None = None,
    baseline: str | None = None,
    run_preset: str | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "task_preset": task_preset,
        "baseline": baseline,
        "run_preset": run_preset,
        "config": config_to_dict(config),
    }


def print_resolved_config(
    config: Any,
    *,
    family: str | None = None,
    task_preset: str | None = None,
    baseline: str | None = None,
    run_preset: str | None = None,
) -> None:
    payload = resolved_config_payload(
        config,
        family=family,
        task_preset=task_preset,
        baseline=baseline,
        run_preset=run_preset,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
