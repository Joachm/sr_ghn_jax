from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiment_configs import MetaBraxConfig
from meta_brax_heading import (
    CARDINAL_HEADINGS,
    ConditionSpec,
    _downsample_frames,
    _heading_label,
    _normalize_video_frames,
    _pipeline_trajectory,
    _side_by_side_frames,
    _write_video_ffmpeg,
    make_runtime_config,
    rollout_individual_on_heading,
    split_support_query_episode_keys,
    sample_heldout_heading_tasks,
    srghn_adapt,
    save_showcase_topdown_plot,
)
from envs import make_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a batch of meta-Brax showcase videos and plots for several headings."
    )
    parser.add_argument("--input", required=True, help="Path to a meta_brax_heading results pickle.")
    parser.add_argument("--condition", default=None, help="Condition name to render. Required if the file has multiple results.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the rendered batch. Defaults to <input-stem>_<condition>_batch.",
    )
    parser.add_argument(
        "--heading-source",
        default="heldout",
        choices=("heldout", "training"),
        help="Choose headings from random held-out directions or the four training cardinals.",
    )
    parser.add_argument("--num-headings", type=int, default=6, help="How many headings to render.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when sampling headings.")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-max-frames", type=int, default=240)
    parser.add_argument("--video-camera", default=None)
    return parser


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


def _sanitize_heading_label(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )


def _sample_render_headings(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    if args.heading_source == "training":
        key = jax.random.PRNGKey(args.seed)
        task_ids = jax.random.randint(key, (args.num_headings,), 0, CARDINAL_HEADINGS.shape[0], dtype=jnp.int32)
        headings = np.asarray(CARDINAL_HEADINGS[task_ids], dtype=np.float32)
        labels = [_heading_label(jnp.asarray(heading, dtype=jnp.float32)) for heading in headings]
        return headings, labels

    sampled = sample_heldout_heading_tasks(jax.random.PRNGKey(args.seed), args.num_headings).headings
    headings = np.asarray(sampled, dtype=np.float32)
    labels = [_heading_label(jnp.asarray(heading, dtype=jnp.float32)) for heading in headings]
    return headings, labels


def _render_single_heading(
    *,
    champion,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    seed: int,
    heading: np.ndarray,
    heading_label: str,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    max_frames: int,
    camera: str | None,
    index: int,
) -> dict[str, object]:
    heading_jax = jnp.asarray(heading, dtype=jnp.float32)
    support_key = jax.random.PRNGKey(seed + 20_000 + index)
    episode_key = jax.random.PRNGKey(seed + 30_000 + index)
    support_keys, query_keys = split_support_query_episode_keys(
        episode_key,
        cfg.support_episodes,
        cfg.query_episodes,
    )
    query_key = query_keys[0]
    before_trajectory, before_return = rollout_individual_on_heading(champion, query_key, heading_jax, cfg)

    adapted = srghn_adapt(champion, support_key, support_keys, heading_jax, cfg, cond)
    after_trajectory, after_return = rollout_individual_on_heading(adapted, query_key, heading_jax, cfg)

    runtime_config = make_runtime_config(cfg)
    env, _, _, _, _, _, _, _ = make_env(runtime_config)
    sys = getattr(env, "sys", None)
    if sys is None:
        raise RuntimeError("Expected Brax environment to expose `env.sys` for rendering.")

    try:
        from brax.io import image as brax_image
    except Exception as exc:
        raise RuntimeError("Brax rendering requires `brax` with `brax.io.image` available.") from exc

    before_frames = _normalize_video_frames(
        brax_image.render_array(
            sys,
            _pipeline_trajectory(before_trajectory),
            height=height,
            width=width,
            camera=camera,
        )
    )
    after_frames = _normalize_video_frames(
        brax_image.render_array(
            sys,
            _pipeline_trajectory(after_trajectory),
            height=height,
            width=width,
            camera=camera,
        )
    )
    before_frames = _downsample_frames(before_frames, max_frames)
    after_frames = _downsample_frames(after_frames, max_frames)
    showcase_frames = _side_by_side_frames(
        before_frames,
        after_frames,
        heading_label=heading_label,
        before_return=before_return,
        after_return=after_return,
    )

    heading_safe = _sanitize_heading_label(heading_label)
    score_safe = _sanitize_heading_label(f"{after_return:.1f}")
    video_path = output_dir / f"{index:02d}_{heading_safe}_score_{score_safe}.mp4"
    plot_path = output_dir / f"{index:02d}_{heading_safe}_score_{score_safe}.png"
    _write_video_ffmpeg(showcase_frames, video_path, fps)
    plot_payload = save_showcase_topdown_plot(
        plot_path,
        heading=heading_jax,
        heading_label=heading_label,
        before_trajectory=before_trajectory,
        after_trajectory=after_trajectory,
        before_return=before_return,
        after_return=after_return,
    )

    return {
        "heading": heading_label,
        "before_return": float(before_return),
        "after_return": float(after_return),
        "improvement": float(after_return - before_return),
        "video_path": str(video_path),
        "plot_path": str(plot_path),
        "plot": plot_payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    condition_name, result = _load_condition_payload(payload, args.condition)
    cfg = MetaBraxConfig(**result["config"])
    cond = ConditionSpec(**result["condition"])
    champion = result["champion"]

    output_dir = Path(args.output_dir) if args.output_dir is not None else input_path.with_name(
        f"{input_path.stem}_{condition_name}_batch"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    headings, labels = _sample_render_headings(args)
    summary = []
    for index, (heading, label) in enumerate(zip(headings, labels)):
        payload = _render_single_heading(
            champion=champion,
            cfg=cfg,
            cond=cond,
            seed=args.seed,
            heading=heading,
            heading_label=label,
            output_dir=output_dir,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
            max_frames=args.video_max_frames,
            camera=args.video_camera,
            index=index,
        )
        summary.append(payload)
        print(
            f"[saved] heading={payload['heading']} "
            f"score={payload['after_return']:.1f} "
            f"video={payload['video_path']} "
            f"plot={payload['plot_path']}"
        )

    summary_path = output_dir / "summary.json"
    try:
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    except Exception:
        pass
    else:
        print(f"[saved] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
