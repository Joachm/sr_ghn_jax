from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from experiment_configs import MetaBraxConfig
from meta_brax_heading import (
    ConditionSpec,
    add_showcase_video_args,
    render_showcase_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a before/after adaptation showcase video from a saved meta-Brax result pickle."
    )
    parser.add_argument("--input", required=True, help="Path to meta_brax_heading results pickle.")
    parser.add_argument("--condition", default=None, help="Condition name to render. Required if the file has multiple results.")
    parser.add_argument("--output", default=None, help="Output mp4 path. Defaults to <input-stem>_<condition>_showcase.mp4.")
    return add_showcase_video_args(parser, include_toggle=False, include_output_dir=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _load_condition_payload(payload: dict, condition_name: str | None) -> tuple[str, dict]:
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("Expected input pickle to contain a non-empty `results` mapping.")
    if condition_name is not None:
        if condition_name not in results:
            raise ValueError(f"Condition {condition_name!r} not found. Available: {sorted(results)}")
        return condition_name, results[condition_name]
    if len(results) == 1:
        only_name = next(iter(results))
        return only_name, results[only_name]
    raise ValueError(f"--condition is required because the file contains multiple results: {sorted(results)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    condition_name, result = _load_condition_payload(payload, args.condition)
    cfg = MetaBraxConfig(**result["config"])
    cond = ConditionSpec(**result["condition"])
    champion = result["champion"]

    output_path = Path(args.output) if args.output is not None else input_path.with_name(
        f"{input_path.stem}_{condition_name}_showcase.mp4"
    )
    showcase = render_showcase_video(
        champion,
        cfg,
        cond,
        output_path,
        heading_choice=args.video_heading,
        width=args.video_width,
        height=args.video_height,
        fps=args.video_fps,
        max_frames=args.video_max_frames,
        camera=args.video_camera,
    )
    print(
        f"[saved] {output_path} "
        f"heading={showcase['heading_label']} "
        f"before={showcase['before_return']:.1f} "
        f"after={showcase['after_return']:.1f} "
        f"improvement={showcase['improvement']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
