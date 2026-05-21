from __future__ import annotations

import argparse
import math
import pickle
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from configs import (
    BASELINE_FIXED_LR,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FULL,
    BASELINE_NO_SELF_REFERENCE,
    baseline_overrides,
    make_config_brax_generic,
)
from envs import make_env
from evolution import init_population
from experiments._common import build_graphs_and_specs
from metrics import compute_experiment_metrics
from obs_norm import normalize_obs
from policy import apply_policy
from srghn import SRGHN, make_policy, mutate_with_metadata, mutation_metadata
from experiment_configs import MetaBraxConfig, build_meta_brax_config, print_resolved_config, resolved_config_payload


CARDINAL_HEADINGS = jnp.asarray(
    (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
    ),
    dtype=jnp.float32,
)

BASELINE_NAMES = (
    BASELINE_FULL,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FIXED_LR,
    BASELINE_NO_SELF_REFERENCE,
)

VELOCITY_KEY_PAIRS = (
    ("x_velocity", "y_velocity"),
    ("velocity_x", "velocity_y"),
    ("x_vel", "y_vel"),
)

POSITION_KEY_PAIRS = (
    ("x_position", "y_position"),
    ("position_x", "position_y"),
)

SHOWCASE_HEADING_CHOICES = ("auto", "training_auto", "pos_x", "neg_x", "pos_y", "neg_y")
SHOWCASE_HEADING_BY_CHOICE = {
    "pos_x": CARDINAL_HEADINGS[0],
    "neg_x": CARDINAL_HEADINGS[1],
    "pos_y": CARDINAL_HEADINGS[2],
    "neg_y": CARDINAL_HEADINGS[3],
}


@jax.tree_util.register_pytree_node_class
@dataclass
class BraxHeadingTaskBatch:
    headings: jnp.ndarray

    def tree_flatten(self):
        return (self.headings,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (headings,) = children
        return cls(headings=headings)


@jax.tree_util.register_pytree_node_class
@dataclass
class SRGHNMetaState:
    pop: SRGHN
    key: jax.Array
    pop_fitness: jnp.ndarray
    best_fitness: jnp.ndarray
    best_indiv: SRGHN

    def tree_flatten(self):
        return (self.pop, self.key, self.pop_fitness, self.best_fitness, self.best_indiv), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        pop, key, pop_fitness, best_fitness, best_indiv = children
        return cls(pop=pop, key=key, pop_fitness=pop_fitness, best_fitness=best_fitness, best_indiv=best_indiv)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    baseline_name: str
    mutation_exclude_modules: tuple[str, ...] = ()
    fixed_mutation_lr: float | None = None


def _safe_wandb_import():
    try:
        import wandb
    except Exception:
        return None
    return wandb


def wandb_config_payload(
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    *,
    policy_spec=None,
    self_spec=None,
) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["condition"] = asdict(cond)
    if policy_spec is not None:
        payload["policy_spec_shapes"] = [tuple(shape) for shape in policy_spec.shapes]
        payload["policy_num_nodes"] = int(policy_spec.num_nodes)
    if self_spec is not None:
        payload["self_spec_shapes"] = [tuple(shape) for shape in self_spec.shapes]
        payload["self_num_nodes"] = int(self_spec.num_nodes)
    return payload


def _wandb_run_name(cfg: MetaBraxConfig, cond: ConditionSpec) -> str:
    prefix = cfg.wandb_name or "meta_brax_heading"
    return f"{prefix}-{cond.name}"


def _wandb_init_run(
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    *,
    policy_spec=None,
    self_spec=None,
):
    if cfg.wandb_project is None:
        return None
    wandb = _safe_wandb_import()
    if wandb is None:
        return None
    try:
        if wandb.run is None:
            return wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                name=_wandb_run_name(cfg, cond),
                config=wandb_config_payload(cfg, cond, policy_spec=policy_spec, self_spec=self_spec),
            )
    except Exception:
        return None
    return wandb.run


def _wandb_log_metrics(metrics: dict[str, Any], step: int, *, prefix: str = "train") -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return
    payload = {f"{prefix}/{key}": float(value) for key, value in metrics.items()}
    payload["gen"] = int(step)
    try:
        wandb.log(payload, step=int(step))
    except Exception:
        pass


def _wandb_log_summary(cfg: MetaBraxConfig, cond: ConditionSpec, payload: dict[str, Any]) -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return

    history = payload["train_history"]
    curve = payload["adaptation_curve_query_return"]
    final_population_fitness = jnp.asarray(payload["final_population_fitness"])
    summary_payload = {
        "final/train_fitness_best": float(history["fitness_best"][-1]),
        "final/train_fitness_mean": float(history["fitness_mean"][-1]),
        "final/train_query_return_best": float(history["query_return_best"][-1]),
        "final/train_query_return_mean": float(history["query_return_mean"][-1]),
        "final/train_query_return_best_so_far": float(history["query_return_best_so_far"][-1]),
        "final/train_diversity": float(history["diversity"][-1]),
        "final/final_population_fitness_best": float(jnp.max(final_population_fitness)),
        "final/final_population_fitness_mean": float(jnp.mean(final_population_fitness)),
        "final/champion_meta_fitness": float(payload["champion_meta_fitness"]),
        "final/heldout_curve_pre_return_mean": float(curve["mean"][0]),
        "final/heldout_curve_post_return_mean": float(curve["mean"][-1]),
        "final/heldout_curve_pre_return_min": float(curve["min"][0]),
        "final/heldout_curve_post_return_min": float(curve["min"][-1]),
        "final/heldout_curve_improvement_min": float(curve["min"][-1] - curve["min"][0]),
        "final/heldout_curve_stderr_post_return": float(curve["stderr"][-1]),
        "final/heldout_curve_improvement_mean": float(curve["mean"][-1] - curve["mean"][0]),
        "final/heldout_curve_length": int(len(curve["mean"])),
    }
    try:
        wandb.log(summary_payload, step=int(cfg.outer_generations))
    except Exception:
        pass

    if not cfg.wandb_log_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["query_return_best"], label="best")
    axes[0].plot(history["query_return_mean"], label="mean")
    axes[0].set_title("Outer meta-train")
    axes[0].set_xlabel("Outer generation")
    axes[0].set_ylabel("Query return")
    axes[0].legend()

    xs = list(range(len(curve["mean"])))
    axes[1].plot(xs, curve["mean"], label="mean")
    axes[1].fill_between(xs, curve["mean"] - curve["stderr"], curve["mean"] + curve["stderr"], alpha=0.2)
    axes[1].set_title("Held-out adaptation")
    axes[1].set_xlabel("Inner generation")
    axes[1].set_ylabel("Query return")
    axes[1].legend()
    fig.suptitle(cond.name)
    fig.tight_layout()

    try:
        wandb.log({"plots/summary": wandb.Image(fig)}, step=int(cfg.outer_generations))
    except Exception:
        pass
    finally:
        plt.close(fig)


def _wandb_finish_run() -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass


def _summary_from_curves(curves: jnp.ndarray) -> dict[str, Any]:
    curves_host = jax.device_get(curves)
    return {
        "all": curves_host,
        "mean": curves_host.mean(axis=0),
        "min": curves_host.min(axis=0),
        "max": curves_host.max(axis=0),
        "std": curves_host.std(axis=0),
        "stderr": curves_host.std(axis=0) / math.sqrt(max(curves_host.shape[0], 1)),
    }


def _history_to_host(history: dict[str, jnp.ndarray]) -> dict[str, Any]:
    return {key: jax.device_get(value) for key, value in history.items()}


def parse_condition_spec(text: str, *, fixed_mutation_lr: float = 0.02) -> ConditionSpec:
    if text not in BASELINE_NAMES:
        raise ValueError(f"Unknown meta-Brax condition '{text}'. Expected one of {BASELINE_NAMES}.")
    overrides = baseline_overrides(text, fixed_mutation_lr=fixed_mutation_lr)
    return ConditionSpec(
        name=text,
        baseline_name=text,
        mutation_exclude_modules=tuple(overrides["mutation_exclude_modules"]),
        fixed_mutation_lr=overrides["fixed_mutation_lr"],
    )


def make_runtime_config(cfg: MetaBraxConfig):
    return make_config_brax_generic(
        cfg.env_id,
        seed=cfg.seed,
        pop_size=cfg.outer_pop_size,
        num_generations=cfg.outer_generations,
        episode_horizon=cfg.episode_horizon,
        children_per_parent=cfg.outer_children_per_parent,
        episodes_per_eval=1,
        brax_backend=cfg.brax_backend,
        parameter_block_size=cfg.parameter_block_size,
        mutation_block_ratio=cfg.mutation_block_ratio,
        policy_hidden_dims=cfg.policy_hidden_dims,
    )


def sample_heading_tasks(key: jax.Array, batch_size: int) -> BraxHeadingTaskBatch:
    angles = jax.random.uniform(
        key,
        (batch_size,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=jnp.float32,
    )
    headings = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    return BraxHeadingTaskBatch(headings=headings)

def sample_heldout_heading_tasks(key: jax.Array, batch_size: int) -> BraxHeadingTaskBatch:
    return sample_heading_tasks(key, batch_size)


def split_support_query_episode_keys(
    key: jax.Array,
    support_episodes: int,
    query_episodes: int,
) -> tuple[jax.Array, jax.Array]:
    total_episodes = support_episodes + query_episodes
    if total_episodes <= 0:
        raise ValueError("support_episodes + query_episodes must be positive.")
    keys = jax.random.split(key, total_episodes)
    return keys[:support_episodes], keys[support_episodes:]


def _metric_mapping(state) -> Any | None:
    metrics = getattr(state, "metrics", None)
    if metrics is None:
        return None
    if hasattr(metrics, "keys"):
        return metrics
    return None


def _extract_planar_velocity_from_metrics(metrics) -> jnp.ndarray | None:
    if metrics is None:
        return None
    for x_key, y_key in VELOCITY_KEY_PAIRS:
        if x_key in metrics and y_key in metrics:
            return jnp.asarray((metrics[x_key], metrics[y_key]), dtype=jnp.float32)
    return None


def _extract_planar_position_from_metrics(metrics) -> jnp.ndarray | None:
    if metrics is None:
        return None
    for x_key, y_key in POSITION_KEY_PAIRS:
        if x_key in metrics and y_key in metrics:
            return jnp.asarray((metrics[x_key], metrics[y_key]), dtype=jnp.float32)
    return None


def _extract_planar_position_from_pipeline_state(pipeline_state) -> jnp.ndarray | None:
    if pipeline_state is None:
        return None
    x = getattr(pipeline_state, "x", None)
    pos = getattr(x, "pos", None)
    if pos is not None:
        pos = jnp.asarray(pos, dtype=jnp.float32)
        if pos.ndim >= 2 and pos.shape[-1] >= 2:
            return pos[0, :2]
    qp = getattr(pipeline_state, "qp", None)
    pos = getattr(qp, "pos", None)
    if pos is not None:
        pos = jnp.asarray(pos, dtype=jnp.float32)
        if pos.ndim >= 2 and pos.shape[-1] >= 2:
            return pos[0, :2]
    return None


def _extract_planar_position_from_state(state) -> jnp.ndarray | None:
    pipeline_state = getattr(state, "pipeline_state", None)
    pipeline_position = _extract_planar_position_from_pipeline_state(pipeline_state)
    if pipeline_position is not None:
        return pipeline_position

    qp = getattr(state, "qp", None)
    pos = getattr(qp, "pos", None)
    if pos is not None:
        pos = jnp.asarray(pos, dtype=jnp.float32)
        if pos.ndim >= 2 and pos.shape[-1] >= 2:
            return pos[0, :2]

    metrics = _metric_mapping(state)
    return _extract_planar_position_from_metrics(metrics)


def _state_attr_names(obj) -> tuple[str, ...]:
    mapping = getattr(obj, "__dict__", None)
    if mapping is None:
        return ()
    return tuple(sorted(mapping.keys()))


def _metric_key_names(metrics) -> tuple[str, ...]:
    if metrics is None or not hasattr(metrics, "keys"):
        return ()
    return tuple(sorted(str(key) for key in metrics.keys()))


def _infer_env_dt(env) -> float:
    if hasattr(env, "dt"):
        return float(env.dt)
    sys = getattr(env, "sys", None)
    config = getattr(sys, "config", None)
    if config is not None and hasattr(config, "dt"):
        return float(config.dt)
    raise ValueError("Unable to infer Brax dt from environment.")


def _extract_planar_velocity(next_state, prev_state, dt: float) -> jnp.ndarray:
    metrics = _metric_mapping(next_state)
    metric_velocity = _extract_planar_velocity_from_metrics(metrics)
    if metric_velocity is not None:
        return metric_velocity

    next_position = _extract_planar_position_from_state(next_state)
    prev_position = _extract_planar_position_from_state(prev_state)
    if next_position is not None and prev_position is not None:
        return (next_position - prev_position) / jnp.asarray(dt, dtype=jnp.float32)

    raise ValueError(
        "Unable to extract planar velocity. Expected either one of the metric key pairs "
        f"{VELOCITY_KEY_PAIRS} or root/torso positions from state.pipeline_state.x.pos, "
        "state.pipeline_state.qp.pos, state.qp.pos, or position metrics "
        f"{POSITION_KEY_PAIRS}. Available metric keys: {_metric_key_names(metrics)}. "
        f"Next-state attrs: {_state_attr_names(next_state)}. Prev-state attrs: {_state_attr_names(prev_state)}."
    )


def _projected_heading_reward(prev_state, next_state, heading: jnp.ndarray, dt: float) -> jnp.ndarray:
    planar_velocity = _extract_planar_velocity(next_state, prev_state, dt)
    return jnp.dot(planar_velocity, jnp.asarray(heading, dtype=jnp.float32))


def evaluate_policy_params_on_heading(
    policy_params,
    episode_keys: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
) -> jnp.ndarray:
    runtime_config = make_runtime_config(cfg)
    env, _, _, _, is_discrete, action_shape, action_low, action_high = make_env(runtime_config)
    if is_discrete:
        raise ValueError("meta_brax_heading only supports continuous Brax environments.")
    dt = _infer_env_dt(env)

    def rollout_once(episode_key: jax.Array) -> jnp.ndarray:
        state = env.reset(episode_key)
        obs = state.obs

        def step_fn(carry, _):
            obs_t, state_t, done_t = carry
            obs_in = normalize_obs(obs_t, None, clip=cfg.obs_norm_clip, eps=cfg.obs_norm_eps)
            action = apply_policy(policy_params, obs_in, runtime_config, is_discrete=False)
            action = jnp.asarray(action, dtype=jnp.float32).reshape(action_shape)
            if action_low is not None and action_high is not None:
                action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)

            def do_step(_):
                next_state = env.step(state_t, action)
                reward = _projected_heading_reward(state_t, next_state, heading, dt)
                done = jnp.asarray(next_state.done, dtype=jnp.bool_)
                return next_state.obs, next_state, reward.astype(jnp.float32), done

            def skip_step(_):
                return obs_t, state_t, jnp.zeros((), dtype=jnp.float32), done_t

            next_obs, next_state, reward, done = jax.lax.cond(done_t, skip_step, do_step, operand=None)
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done), reward

        (_, _, _), rewards = jax.lax.scan(
            step_fn,
            (obs, state, jnp.asarray(False)),
            None,
            length=cfg.episode_horizon,
        )
        return jnp.sum(rewards)

    returns = jax.vmap(rollout_once)(episode_keys)
    return jnp.mean(returns)


def evaluate_individual_on_heading(
    indiv: SRGHN,
    episode_keys: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
) -> jnp.ndarray:
    return evaluate_policy_params_on_heading(make_policy(indiv), episode_keys, heading, cfg)


def _heading_label(heading: jnp.ndarray) -> str:
    heading_np = tuple(float(value) for value in np.asarray(jax.device_get(heading)).tolist())
    mapping = {
        (1.0, 0.0): "+x",
        (-1.0, 0.0): "-x",
        (0.0, 1.0): "+y",
        (0.0, -1.0): "-y",
    }
    return mapping.get(heading_np, str(heading_np))


def _showcase_heading_from_choice(choice: str) -> jnp.ndarray:
    if choice not in SHOWCASE_HEADING_BY_CHOICE:
        raise ValueError(f"Unknown showcase heading choice: {choice}")
    return jnp.asarray(SHOWCASE_HEADING_BY_CHOICE[choice], dtype=jnp.float32)


def _showcase_key_for_index(seed: int, index: int) -> jax.Array:
    return jax.random.PRNGKey(seed + 30_000 + 997 * index)


def _showcase_adapt_key_for_index(seed: int, index: int) -> jax.Array:
    return jax.random.PRNGKey(seed + 40_000 + 997 * index)


def rollout_policy_params_on_heading(
    policy_params,
    episode_key: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
) -> tuple[list[Any], float]:
    runtime_config = make_runtime_config(cfg)
    env, _, _, _, is_discrete, action_shape, action_low, action_high = make_env(runtime_config)
    if is_discrete:
        raise ValueError("meta_brax_heading only supports continuous Brax environments.")
    dt = _infer_env_dt(env)

    def step_once(state_t):
        obs_t = state_t.obs
        obs_in = normalize_obs(obs_t, None, clip=cfg.obs_norm_clip, eps=cfg.obs_norm_eps)
        action = apply_policy(policy_params, obs_in, runtime_config, is_discrete=False)
        action = jnp.asarray(action, dtype=jnp.float32).reshape(action_shape)
        if action_low is not None and action_high is not None:
            action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
        next_state = env.step(state_t, action)
        reward = _projected_heading_reward(state_t, next_state, heading, dt)
        done = jnp.asarray(next_state.done, dtype=jnp.bool_)
        return next_state, reward.astype(jnp.float32), done

    step_once_jit = jax.jit(step_once)

    state = env.reset(episode_key)
    trajectory = [jax.device_get(state)]
    total_reward = 0.0
    done = False

    for _ in range(cfg.episode_horizon):
        if done:
            trajectory.append(jax.device_get(state))
            continue
        state, reward, done = step_once_jit(state)
        total_reward += float(jax.device_get(reward))
        done = bool(jax.device_get(done))
        trajectory.append(jax.device_get(state))

    return trajectory, total_reward


def rollout_individual_on_heading(
    indiv: SRGHN,
    episode_key: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
) -> tuple[list[Any], float]:
    return rollout_policy_params_on_heading(make_policy(indiv), episode_key, heading, cfg)


def _pipeline_trajectory(trajectory: list[Any]) -> list[Any]:
    pipeline_states = []
    for state in trajectory:
        pipeline_state = getattr(state, "pipeline_state", None)
        if pipeline_state is not None:
            pipeline_states.append(pipeline_state)
            continue
        qp = getattr(state, "qp", None)
        if qp is not None:
            pipeline_states.append(qp)
            continue
        raise ValueError("Expected Brax env state to expose `.pipeline_state` or `.qp` for rendering.")
    return pipeline_states


def _normalize_video_frames(frames) -> np.ndarray:
    if isinstance(frames, (list, tuple)):
        frames = np.stack([np.asarray(frame) for frame in frames], axis=0)
    else:
        frames = np.asarray(frames)

    if frames.ndim != 4:
        raise ValueError(f"Expected rendered frames to have rank 4, got shape {frames.shape}.")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.dtype == np.uint8:
        return frames
    if np.issubdtype(frames.dtype, np.floating):
        max_val = float(np.max(frames)) if frames.size else 0.0
        if max_val <= 1.0:
            frames = frames * 255.0
    return np.clip(frames, 0, 255).astype(np.uint8)


def _downsample_frames(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or frames.shape[0] <= max_frames:
        return frames
    indices = np.linspace(0, frames.shape[0] - 1, num=max_frames, dtype=np.int32)
    return frames[indices]


def _side_by_side_frames(
    before_frames: np.ndarray,
    after_frames: np.ndarray,
    *,
    heading_label: str,
    before_return: float,
    after_return: float,
) -> np.ndarray:
    frame_count = min(before_frames.shape[0], after_frames.shape[0])
    before_frames = before_frames[:frame_count]
    after_frames = after_frames[:frame_count]
    separator = np.full((frame_count, before_frames.shape[1], 6, 3), 255, dtype=np.uint8)
    combined = np.concatenate([before_frames, separator, after_frames], axis=2)

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return combined

    banner_height = 40
    banner = Image.new("RGB", (combined.shape[2], banner_height), (18, 18, 18))
    draw = ImageDraw.Draw(banner)
    draw.text((12, 12), f"Before adaptation  return={before_return:.1f}", fill=(255, 255, 255))
    draw.text(
        (before_frames.shape[2] + separator.shape[2] + 12, 12),
        f"After adaptation  return={after_return:.1f}",
        fill=(255, 255, 255),
    )
    heading_text = f"Hidden heading: {heading_label}"
    if hasattr(draw, "textbbox"):
        left, _, right, _ = draw.textbbox((0, 0), heading_text)
        heading_x = max((combined.shape[2] - (right - left)) // 2, 12)
    else:
        heading_x = max(combined.shape[2] // 2 - 60, 12)
    draw.text((heading_x, 12), heading_text, fill=(180, 220, 255))

    banner_arr = np.asarray(banner, dtype=np.uint8)
    banner_arr = np.repeat(banner_arr[None, ...], frame_count, axis=0)
    return np.concatenate([banner_arr, combined], axis=1)


def _write_video_ffmpeg(frames: np.ndarray, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write mp4 output but was not found on PATH.")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape (T, H, W, 3), got {frames.shape}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _, stderr = proc.communicate(frames.tobytes())
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace") or "ffmpeg failed to encode video.")


def _trajectory_positions_xy(trajectory: list[Any]) -> np.ndarray:
    positions: list[np.ndarray] = []
    for state in trajectory:
        position = _extract_planar_position_from_state(state)
        if position is None:
            raise ValueError("Unable to extract planar position for top-down trajectory plotting.")
        positions.append(np.asarray(jax.device_get(position), dtype=np.float32))
    return np.stack(positions, axis=0)


def save_showcase_topdown_plot(
    output_path: Path,
    *,
    heading: jnp.ndarray,
    heading_label: str,
    before_trajectory: list[Any],
    after_trajectory: list[Any],
    before_return: float,
    after_return: float,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to save the top-down trajectory plot.") from exc

    before_xy = _trajectory_positions_xy(before_trajectory)
    after_xy = _trajectory_positions_xy(after_trajectory)
    heading_xy = np.asarray(jax.device_get(jnp.asarray(heading, dtype=jnp.float32)), dtype=np.float32)

    all_xy = np.concatenate([before_xy, after_xy], axis=0)
    origin = before_xy[0]
    max_extent = float(np.max(np.abs(all_xy - origin))) if all_xy.size else 1.0
    arrow_scale = max(max_extent * 0.6, 1.0)
    heading_arrow = heading_xy * arrow_scale

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(before_xy[:, 0], before_xy[:, 1], color="#d95f02", linewidth=2.0, label=f"Before ({before_return:.1f})")
    ax.plot(after_xy[:, 0], after_xy[:, 1], color="#1b9e77", linewidth=2.0, label=f"After ({after_return:.1f})")

    ax.scatter(before_xy[0, 0], before_xy[0, 1], color="black", s=50, marker="o", label="Start")
    ax.scatter(before_xy[-1, 0], before_xy[-1, 1], color="#d95f02", s=60, marker="x", label="Before end")
    ax.scatter(after_xy[-1, 0], after_xy[-1, 1], color="#1b9e77", s=60, marker="x", label="After end")

    ax.arrow(
        origin[0],
        origin[1],
        heading_arrow[0],
        heading_arrow[1],
        width=max(arrow_scale * 0.01, 0.01),
        head_width=max(arrow_scale * 0.08, 0.08),
        head_length=max(arrow_scale * 0.12, 0.12),
        length_includes_head=True,
        color="#377eb8",
        alpha=0.9,
    )
    ax.text(
        origin[0] + heading_arrow[0],
        origin[1] + heading_arrow[1],
        f" goal {heading_label}",
        color="#377eb8",
        fontsize=10,
        va="bottom",
    )

    ax.set_title(f"Meta-Brax showcase top-down view\nGoal direction {heading_label}")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    padding = max(max_extent * 0.15, 0.5)
    ax.set_xlim(float(np.min(all_xy[:, 0])) - padding, float(np.max(all_xy[:, 0])) + padding)
    ax.set_ylim(float(np.min(all_xy[:, 1])) - padding, float(np.max(all_xy[:, 1])) + padding)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "before_start": before_xy[0].tolist(),
        "before_end": before_xy[-1].tolist(),
        "after_end": after_xy[-1].tolist(),
    }


def choose_showcase_episode(
    indiv: SRGHN,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    *,
    heading_choice: str = "auto",
) -> dict[str, Any]:
    query_episodes = max(int(cfg.query_episodes), 1)
    if heading_choice == "auto":
        auto_tasks = sample_heldout_heading_tasks(jax.random.PRNGKey(cfg.seed + 50_000), 8)
        candidate_items = tuple((f"auto_{idx}", heading) for idx, heading in enumerate(auto_tasks.headings))
    elif heading_choice == "training_auto":
        auto_tasks = sample_heading_tasks(jax.random.PRNGKey(cfg.seed + 60_000), 8)
        candidate_items = tuple((f"train_{idx}", heading) for idx, heading in enumerate(auto_tasks.headings))
    else:
        candidate_items = ((heading_choice, _showcase_heading_from_choice(heading_choice)),)
    best_payload: dict[str, Any] | None = None

    for idx, (choice_name, heading) in enumerate(candidate_items):
        episode_key = _showcase_key_for_index(cfg.seed, idx)
        support_keys, query_keys = split_support_query_episode_keys(
            episode_key,
            cfg.support_episodes,
            query_episodes,
        )
        query_key = query_keys[0]
        before_return = float(jax.device_get(evaluate_individual_on_heading(indiv, query_keys[:1], heading, cfg)))
        if cfg.inner_generations > 0 and cfg.support_episodes > 0:
            adapted = srghn_adapt(
                indiv,
                _showcase_adapt_key_for_index(cfg.seed, idx),
                support_keys,
                heading,
                cfg,
                cond,
            )
        else:
            adapted = indiv
        after_return = float(jax.device_get(evaluate_individual_on_heading(adapted, query_keys[:1], heading, cfg)))
        payload = {
            "choice": choice_name,
            "heading": jax.device_get(heading),
            "heading_label": _heading_label(heading),
            "support_keys": support_keys,
            "query_key": query_key,
            "before_return": before_return,
            "after_return": after_return,
            "improvement": after_return - before_return,
            "adapted": adapted,
        }
        if best_payload is None or payload["improvement"] > best_payload["improvement"]:
            best_payload = payload

    if best_payload is None:
        raise RuntimeError("No showcase heading candidates were evaluated.")
    return best_payload


def render_showcase_video(
    indiv: SRGHN,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    output_path: Path,
    *,
    heading_choice: str = "auto",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    max_frames: int = 240,
    camera: str | None = None,
) -> dict[str, Any]:
    payload = render_showcase_artifacts(
        indiv,
        cfg,
        cond,
        output_path,
        heading_choice=heading_choice,
        width=width,
        height=height,
        fps=fps,
        max_frames=max_frames,
        camera=camera,
        plot_output_path=output_path.with_suffix(".png"),
    )
    return {
        "output_path": payload["video_output_path"],
        "plot_output_path": payload["plot_output_path"],
        "heading_choice": payload["heading_choice"],
        "heading_label": payload["heading_label"],
        "before_return": payload["before_return"],
        "after_return": payload["after_return"],
        "improvement": payload["improvement"],
        "num_frames": payload["num_frames"],
        "fps": payload["fps"],
    }


def render_showcase_artifacts(
    indiv: SRGHN,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    video_output_path: Path,
    *,
    heading_choice: str = "auto",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    max_frames: int = 240,
    camera: str | None = None,
    plot_output_path: Path | None = None,
) -> dict[str, Any]:
    try:
        from brax.io import image as brax_image
    except Exception as exc:
        raise RuntimeError("Brax rendering requires `brax` with `brax.io.image` available.") from exc

    showcase = choose_showcase_episode(indiv, cfg, cond, heading_choice=heading_choice)
    heading = jnp.asarray(showcase["heading"], dtype=jnp.float32)
    before_trajectory, before_return = rollout_individual_on_heading(indiv, showcase["query_key"], heading, cfg)
    after_trajectory, after_return = rollout_individual_on_heading(showcase["adapted"], showcase["query_key"], heading, cfg)

    runtime_config = make_runtime_config(cfg)
    env, _, _, _, _, _, _, _ = make_env(runtime_config)
    sys = getattr(env, "sys", None)
    if sys is None:
        raise RuntimeError("Expected Brax environment to expose `env.sys` for rendering.")

    before_frames = _normalize_video_frames(
        brax_image.render_array(sys, _pipeline_trajectory(before_trajectory), height=height, width=width, camera=camera)
    )
    after_frames = _normalize_video_frames(
        brax_image.render_array(sys, _pipeline_trajectory(after_trajectory), height=height, width=width, camera=camera)
    )
    before_frames = _downsample_frames(before_frames, max_frames)
    after_frames = _downsample_frames(after_frames, max_frames)
    showcase_frames = _side_by_side_frames(
        before_frames,
        after_frames,
        heading_label=showcase["heading_label"],
        before_return=before_return,
        after_return=after_return,
    )
    _write_video_ffmpeg(showcase_frames, video_output_path, fps)

    resolved_plot_output = plot_output_path or video_output_path.with_suffix(".png")
    plot_payload = save_showcase_topdown_plot(
        resolved_plot_output,
        heading=heading,
        heading_label=showcase["heading_label"],
        before_trajectory=before_trajectory,
        after_trajectory=after_trajectory,
        before_return=before_return,
        after_return=after_return,
    )

    return {
        "video_output_path": str(video_output_path),
        "plot_output_path": str(resolved_plot_output),
        "heading_choice": showcase["choice"],
        "heading_label": showcase["heading_label"],
        "before_return": before_return,
        "after_return": after_return,
        "improvement": after_return - before_return,
        "num_frames": int(showcase_frames.shape[0]),
        "fps": int(fps),
        "plot": plot_payload,
    }


def _stack_parent_with_children(parent: SRGHN, children: SRGHN) -> SRGHN:
    parent_arr, _ = eqx.partition(parent, eqx.is_array)
    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p[None, ...], c], axis=0), parent_arr, child_arr)
    return eqx.combine(all_arr, child_static)


def _select_srghn(pop: SRGHN, idx: int | jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    indiv_arr = jax.tree_util.tree_map(lambda value: value[idx], pop_arr)
    return eqx.combine(indiv_arr, pop_static)


def _repeat_srghn(indiv: SRGHN, repeats: int) -> SRGHN:
    arr, static = eqx.partition(indiv, eqx.is_array)
    repeated = jax.tree_util.tree_map(lambda value: jnp.broadcast_to(value, (repeats,) + value.shape), arr)
    return eqx.combine(repeated, static)


def _select_batch_srghn(pop: SRGHN, idx: jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    batch_arr = jax.tree_util.tree_map(lambda value: value[idx], pop_arr)
    return eqx.combine(batch_arr, pop_static)


def _srghn_mutation_kwargs(cond: ConditionSpec) -> dict[str, Any]:
    return {
        "excluded_modules": cond.mutation_exclude_modules,
        "fixed_mutation_lr": cond.fixed_mutation_lr,
    }


def srghn_population_generation(
    pop: SRGHN,
    key: jax.Array,
    pop_size: int,
    children_per_parent: int,
    fitness_fn,
    cond: ConditionSpec,
):
    mutation_kwargs = _srghn_mutation_kwargs(cond)
    probe_key, child_key = jax.random.split(key)
    probe_keys = jax.random.split(probe_key, pop_size)
    parent_metadata = eqx.filter_vmap(lambda indiv, rng: mutation_metadata(indiv, rng, **mutation_kwargs))(pop, probe_keys)

    if children_per_parent <= 0:
        all_candidates = pop
        all_fitness = eqx.filter_vmap(fitness_fn)(all_candidates)
        select_idx = jnp.argsort(all_fitness)[-pop_size:]
        next_pop = _select_batch_srghn(all_candidates, select_idx)
        next_fitness = all_fitness[select_idx]
        elite_metadata = jax.tree_util.tree_map(lambda value: value[select_idx], parent_metadata)
        metrics = compute_experiment_metrics(pop, all_fitness, parent_metadata, elite_metadata)
        return next_pop, next_fitness, metrics

    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(pop_size), children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda value: value[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(child_key, pop_size * children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, rng: mutate_with_metadata(indiv, rng, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda parent, child: jnp.concatenate([parent, child], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)
    all_fitness = eqx.filter_vmap(fitness_fn)(all_candidates)

    select_idx = jnp.argsort(all_fitness)[-pop_size:]
    next_arr = jax.tree_util.tree_map(lambda value: value[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, child_static)
    next_fitness = all_fitness[select_idx]

    candidate_metadata = jax.tree_util.tree_map(
        lambda parent_value, child_value: jnp.concatenate([parent_value, child_value], axis=0),
        parent_metadata,
        child_metadata,
    )
    elite_metadata = jax.tree_util.tree_map(lambda value: value[select_idx], candidate_metadata)
    metrics = compute_experiment_metrics(pop, all_fitness, parent_metadata, elite_metadata)
    return next_pop, next_fitness, metrics


def init_srghn_local_population(indiv: SRGHN, key: jax.Array, cfg: MetaBraxConfig, cond: ConditionSpec) -> SRGHN:
    if cfg.inner_pop_size <= 1:
        return _repeat_srghn(indiv, 1)
    mutation_kwargs = _srghn_mutation_kwargs(cond)
    child_keys = jax.random.split(key, cfg.inner_pop_size - 1)
    children, _ = eqx.filter_vmap(lambda rng: mutate_with_metadata(indiv, rng, **mutation_kwargs))(child_keys)
    return _stack_parent_with_children(indiv, children)


def srghn_adapt(
    indiv: SRGHN,
    key: jax.Array,
    support_keys: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
) -> SRGHN:
    key_init, key_loop = jax.random.split(key)
    init_pop = init_srghn_local_population(indiv, key_init, cfg, cond)

    def support_fitness(candidate: SRGHN) -> jnp.ndarray:
        return evaluate_individual_on_heading(candidate, support_keys, heading, cfg)

    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = srghn_population_generation(
            pop,
            step_key,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
            cond,
        )
        best_indiv = _select_srghn(next_pop, jnp.argmax(next_fitness))
        return (next_pop, rng, next_fitness), best_indiv

    if cfg.inner_generations <= 0:
        return _select_srghn(init_pop, jnp.argmax(init_fitness))

    (_, _, _), best_seq = jax.lax.scan(
        step_fn,
        (init_pop, key_loop, init_fitness),
        None,
        length=cfg.inner_generations,
    )
    return _select_srghn(best_seq, -1)


def srghn_task_curve(
    indiv: SRGHN,
    key: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
) -> jnp.ndarray:
    key_adapt, key_episode = jax.random.split(key)
    support_keys, query_keys = split_support_query_episode_keys(key_episode, cfg.support_episodes, cfg.query_episodes)
    key_init, key_loop = jax.random.split(key_adapt)
    init_pop = init_srghn_local_population(indiv, key_init, cfg, cond)

    def support_fitness(candidate: SRGHN) -> jnp.ndarray:
        return evaluate_individual_on_heading(candidate, support_keys, heading, cfg)

    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)
    best0 = _select_srghn(init_pop, jnp.argmax(init_fitness))
    initial_return = evaluate_individual_on_heading(best0, query_keys, heading, cfg)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = srghn_population_generation(
            pop,
            step_key,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
            cond,
        )
        best_indiv = _select_srghn(next_pop, jnp.argmax(next_fitness))
        query_return = evaluate_individual_on_heading(best_indiv, query_keys, heading, cfg)
        return (next_pop, rng, next_fitness), query_return

    if cfg.inner_generations <= 0:
        return initial_return[None]

    (_, _, _), query_returns = jax.lax.scan(
        step_fn,
        (init_pop, key_loop, init_fitness),
        None,
        length=cfg.inner_generations,
    )
    return jnp.concatenate([initial_return[None], query_returns], axis=0)


def srghn_meta_fitness(
    indiv: SRGHN,
    key: jax.Array,
    tasks: BraxHeadingTaskBatch,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
) -> jnp.ndarray:
    task_keys = jax.random.split(key, tasks.headings.shape[0])

    def per_task(heading, task_key):
        key_adapt, key_episode = jax.random.split(task_key)
        support_keys, query_keys = split_support_query_episode_keys(key_episode, cfg.support_episodes, cfg.query_episodes)
        adapted = srghn_adapt(indiv, key_adapt, support_keys, heading, cfg, cond)
        return evaluate_individual_on_heading(adapted, query_keys, heading, cfg)

    return jnp.mean(jax.vmap(per_task)(tasks.headings, task_keys))


def srghn_outer_step(state: SRGHNMetaState, gen: jnp.ndarray, cfg: MetaBraxConfig, cond: ConditionSpec):
    key_next, key_tasks, key_eval, key_evolve = jax.random.split(state.key, 4)
    del key_tasks
    tasks = sample_heading_tasks(key_tasks, cfg.meta_batch_size)
    eval_keys = jax.random.split(key_eval, cfg.outer_pop_size * (1 + cfg.outer_children_per_parent))

    def batched_fitness(indiv: SRGHN, eval_key: jax.Array) -> jnp.ndarray:
        return srghn_meta_fitness(indiv, eval_key, tasks, cfg, cond)

    mutation_kwargs = _srghn_mutation_kwargs(cond)
    probe_key, child_key = jax.random.split(key_evolve)
    probe_keys = jax.random.split(probe_key, cfg.outer_pop_size)
    parent_metadata = eqx.filter_vmap(
        lambda indiv, rng: mutation_metadata(indiv, rng, **mutation_kwargs)
    )(state.pop, probe_keys)

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(cfg.outer_pop_size), cfg.outer_children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda value: value[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(child_key, cfg.outer_pop_size * cfg.outer_children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, rng: mutate_with_metadata(indiv, rng, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda parent, child: jnp.concatenate([parent, child], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)
    all_fitness = eqx.filter_vmap(batched_fitness)(all_candidates, eval_keys)

    select_idx = jnp.argsort(all_fitness)[-cfg.outer_pop_size:]
    next_arr = jax.tree_util.tree_map(lambda value: value[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, child_static)
    next_fitness = all_fitness[select_idx]

    candidate_metadata = jax.tree_util.tree_map(
        lambda parent_value, child_value: jnp.concatenate([parent_value, child_value], axis=0),
        parent_metadata,
        child_metadata,
    )
    elite_metadata = jax.tree_util.tree_map(lambda value: value[select_idx], candidate_metadata)

    gen_best_idx = jnp.argmax(next_fitness)
    gen_best = _select_srghn(next_pop, gen_best_idx)
    gen_best_fit = next_fitness[gen_best_idx]

    metrics = compute_experiment_metrics(state.pop, all_fitness, parent_metadata, elite_metadata)
    metrics["query_return_best"] = metrics["fitness_best"]
    metrics["query_return_mean"] = metrics["fitness_mean"]
    metrics["query_return_best_so_far"] = jnp.maximum(state.best_fitness, gen_best_fit)
    jax.debug.callback(partial(_wandb_log_metrics, prefix="train"), metrics, gen)
    improved = gen_best_fit > state.best_fitness
    best_indiv = jax.lax.cond(improved, lambda _: gen_best, lambda _: state.best_indiv, operand=None)
    best_fitness = jnp.maximum(state.best_fitness, gen_best_fit)

    return SRGHNMetaState(
        pop=next_pop,
        key=key_next,
        pop_fitness=next_fitness,
        best_fitness=best_fitness,
        best_indiv=best_indiv,
    ), metrics


@partial(jax.jit, static_argnums=(0, 1))
def run_srghn_compiled(cfg: MetaBraxConfig, cond: ConditionSpec, key: jax.Array):
    runtime_config = make_runtime_config(cfg)
    graphs, specs = build_graphs_and_specs(runtime_config)

    key_pop, key_loop = jax.random.split(key)
    init_pop = init_population(key_pop, runtime_config, graphs, specs)
    init_best = _select_srghn(init_pop, 0)
    init_state = SRGHNMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.outer_pop_size,), dtype=jnp.float32),
        best_fitness=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_indiv=init_best,
    )

    generations = jnp.arange(cfg.outer_generations, dtype=jnp.int32)
    final_state, history = jax.lax.scan(
        lambda carry, gen: srghn_outer_step(carry, gen, cfg, cond),
        init_state,
        generations,
    )

    heldout_tasks = sample_heldout_heading_tasks(jax.random.PRNGKey(cfg.seed + 10_000), cfg.heldout_task_batch_size)
    heldout_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 20_000), cfg.heldout_task_batch_size)

    curves = jax.vmap(
        lambda heading, task_key: srghn_task_curve(final_state.best_indiv, task_key, heading, cfg, cond)
    )(heldout_tasks.headings, heldout_keys)
    return final_state, history, curves


def run_condition(cfg: MetaBraxConfig, cond: ConditionSpec) -> dict[str, Any]:
    runtime_config = make_runtime_config(cfg)
    graphs, specs = build_graphs_and_specs(runtime_config)
    del graphs

    _wandb_init_run(cfg, cond, policy_spec=specs.policy_spec, self_spec=specs.self_spec)

    init_key = jax.random.PRNGKey(cfg.seed)
    t0 = time.perf_counter()
    final_state, history, curves = run_srghn_compiled(cfg, cond, init_key)
    jax.block_until_ready(history["fitness_best"])
    jax.block_until_ready(curves)
    train_seconds = time.perf_counter() - t0
    eval_seconds = 0.0
    champion = final_state.best_indiv

    summary_payload = {
        "train_history": _history_to_host(history),
        "adaptation_curve_query_return": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
        "champion_meta_fitness": float(jax.device_get(final_state.best_fitness)),
    }
    _wandb_log_summary(cfg, cond, summary_payload)
    _wandb_finish_run()

    return {
        "kind": "srghn",
        "condition": asdict(cond),
        "config": asdict(cfg),
        "policy_spec_shapes": [tuple(shape) for shape in specs.policy_spec.shapes],
        "self_spec_shapes": [tuple(shape) for shape in specs.self_spec.shapes],
        "train_history": _history_to_host(history),
        "adaptation_curve_query_return": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
        "champion_meta_fitness": float(jax.device_get(final_state.best_fitness)),
        "champion": champion,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }


def maybe_save_plots(output_stem: Path, results: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, payload in results.items():
        history = payload["train_history"]
        ax.plot(history["query_return_best"], label=name)
    ax.set_title("Meta-train best query return")
    ax.set_xlabel("Outer generation")
    ax.set_ylabel("Query return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_train.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, payload in results.items():
        curve = payload["adaptation_curve_query_return"]
        xs = list(range(len(curve["mean"])))
        mean = curve["mean"]
        stderr = curve["stderr"]
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    ax.set_title("Held-out adaptation curve")
    ax.set_xlabel("Inner generation")
    ax.set_ylabel("Query return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_adaptation.png"), dpi=180)
    plt.close(fig)


def add_showcase_video_args(
    parser: argparse.ArgumentParser,
    *,
    include_toggle: bool = True,
    include_output_dir: bool = True,
) -> argparse.ArgumentParser:
    if include_toggle:
        parser.add_argument("--video", action="store_true", help="Render a before/after adaptation showcase mp4 after training.")
    if include_output_dir:
        parser.add_argument("--video-output-dir", default=None, help="Directory for showcase videos. Defaults near --output.")
    parser.add_argument("--video-heading", default="auto", choices=SHOWCASE_HEADING_CHOICES)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-max-frames", type=int, default=240)
    parser.add_argument("--video-camera", default=None)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta-RL benchmark for Brax Ant with hidden headings.")
    parser.add_argument("--conditions", nargs="+", default=BASELINE_NAMES)
    parser.add_argument("--output", default="meta_brax_heading_results.pkl")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-preset", default="default", choices=("default", "fast"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--outer-generations", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--heldout-task-batch-size", type=int, default=None)
    parser.add_argument("--outer-pop-size", type=int, default=None)
    parser.add_argument("--outer-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-pop-size", type=int, default=None)
    parser.add_argument("--inner-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-generations", type=int, default=None)
    parser.add_argument("--support-episodes", type=int, default=None)
    parser.add_argument("--query-episodes", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--policy-hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--gnn-hidden-dim", type=int, default=None)
    parser.add_argument("--gnn-steps-policy", type=int, default=None)
    parser.add_argument("--gnn-steps-self", type=int, default=None)
    parser.add_argument("--stoch-coeff-dim", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    parser.add_argument("--mutation-rate-head-dim", type=int, default=None)
    parser.add_argument("--const-noise-std", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-plots", action="store_true")
    return add_showcase_video_args(parser)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def make_base_cfg(args: argparse.Namespace) -> MetaBraxConfig:
    overrides = {
        key: value
        for key, value in {
            "env_id": args.env_id,
            "brax_backend": args.backend,
            "seed": args.seed,
            "outer_generations": args.outer_generations,
            "meta_batch_size": args.meta_batch_size,
            "heldout_task_batch_size": args.heldout_task_batch_size,
            "outer_pop_size": args.outer_pop_size,
            "outer_children_per_parent": args.outer_children_per_parent,
            "inner_pop_size": args.inner_pop_size,
            "inner_children_per_parent": args.inner_children_per_parent,
            "inner_generations": args.inner_generations,
            "support_episodes": args.support_episodes,
            "query_episodes": args.query_episodes,
            "episode_horizon": args.episode_horizon,
            "policy_hidden_dims": None if args.policy_hidden_dims is None else tuple(args.policy_hidden_dims),
            "embedding_dim": args.embedding_dim,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "gnn_steps_policy": args.gnn_steps_policy,
            "gnn_steps_self": args.gnn_steps_self,
            "stoch_coeff_dim": args.stoch_coeff_dim,
            "parameter_block_size": args.parameter_block_size,
            "mutation_block_ratio": args.mutation_block_ratio,
            "mutation_rate_head_dim": args.mutation_rate_head_dim,
            "const_noise_std": args.const_noise_std,
            "baseline_fixed_mutation_lr": args.fixed_mutation_lr,
            "wandb_project": None if args.no_wandb else args.wandb_project,
            "wandb_group": args.wandb_group,
            "wandb_name": args.wandb_name,
            "wandb_log_plots": args.wandb_log_plots,
        }.items()
        if value is not None
    }
    if args.no_wandb:
        overrides["wandb_project"] = None
    run_preset = "fast" if args.fast else args.run_preset
    return build_meta_brax_config(run_preset=run_preset, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = make_base_cfg(args)
    if args.print_config:
        print_resolved_config(
            base_cfg,
            family="meta_brax_heading",
            run_preset="fast" if args.fast else args.run_preset,
        )
        return 0
    conditions = [
        parse_condition_spec(text, fixed_mutation_lr=base_cfg.baseline_fixed_mutation_lr)
        for text in args.conditions
    ]

    results: dict[str, Any] = {}
    for cond in conditions:
        print(f"[run] {cond.name} env={base_cfg.env_id} backend={base_cfg.brax_backend}", flush=True)
        payload = run_condition(base_cfg, cond)
        results[cond.name] = payload
        curve = payload["adaptation_curve_query_return"]["mean"]
        print(
            f"[done] {cond.name} train_s={payload['train_seconds']:.2f} eval_s={payload['eval_seconds']:.2f} "
            f"heldout_return={curve.tolist()}",
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_brax_heading",
            run_preset="fast" if args.fast else args.run_preset,
        ),
        "conditions": [asdict(cond) for cond in conditions],
        "results": results,
    }
    with output_path.open("wb") as handle:
        pickle.dump(save_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}")

    if args.video:
        video_dir = Path(args.video_output_dir) if args.video_output_dir is not None else output_path.with_suffix("")
        video_dir.mkdir(parents=True, exist_ok=True)
        for cond in conditions:
            payload = results[cond.name]
            video_path = video_dir / f"{output_path.stem}_{cond.name}_showcase.mp4"
            try:
                showcase = render_showcase_video(
                    payload["champion"],
                    base_cfg,
                    cond,
                    video_path,
                    heading_choice=args.video_heading,
                    width=args.video_width,
                    height=args.video_height,
                    fps=args.video_fps,
                    max_frames=args.video_max_frames,
                    camera=args.video_camera,
                )
                print(
                    "[saved] showcase video "
                    f"{video_path} plot={showcase['plot_output_path']} heading={showcase['heading_label']} "
                    f"before={showcase['before_return']:.1f} after={showcase['after_return']:.1f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[warn] failed to render showcase video for {cond.name}: {exc}", flush=True)

    if args.plot:
        maybe_save_plots(output_path.with_suffix(""), results)
        print(f"[saved] plots near {output_path.with_suffix('')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
