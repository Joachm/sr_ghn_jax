from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env, map_action_for_switch
from obs_norm import normalize_obs
from policy import apply_policy
from solution_artifacts import load_solution_artifact
from srghn import make_policy


def _normalize_frames(frames) -> np.ndarray:
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
        stdout, stderr = proc.communicate(frames.tobytes())
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace") or "ffmpeg failed to encode video.")


def _rollout_trajectory(individual, config, *, key: jax.random.KeyArray, obs_norm_state=None):
    if config.env_backend != "mujoco_playground":
        raise ValueError(
            f"Replay script only supports mujoco_playground artifacts, got env_backend={config.env_backend!r}."
        )

    env, _, _, _, is_discrete, action_shape, action_low, action_high = make_env(config)
    policy_params = make_policy(individual)
    state = env.reset(key)
    trajectory = [state]
    total_reward = 0.0
    gen = jnp.asarray(config.num_generations - 1, dtype=jnp.int32)

    for _ in range(config.episode_horizon):
        obs = state.obs
        obs_in = normalize_obs(obs, obs_norm_state, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
        action = apply_policy(policy_params, obs_in, is_discrete=is_discrete)
        action = map_action_for_switch(action, gen, config)
        if is_discrete:
            action = jnp.asarray(action, dtype=jnp.int32)
        else:
            action = action.reshape(action_shape)
            if action_low is not None:
                action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)

        state = env.step(state, action)
        trajectory.append(state)
        total_reward += float(jnp.asarray(state.reward, dtype=jnp.float32))
        if bool(jnp.asarray(state.done)):
            break

    return env, trajectory, total_reward


def _render_frames(env, trajectory, *, width: int, height: int, camera: str | None):
    if not hasattr(env, "render"):
        raise RuntimeError("Environment does not expose env.render(trajectory, ...).")

    trajectory_candidates = [trajectory]
    if trajectory and all(hasattr(state, "pipeline_state") for state in trajectory):
        trajectory_candidates.append([state.pipeline_state for state in trajectory])
    if trajectory and all(hasattr(state, "data") for state in trajectory):
        trajectory_candidates.append([state.data for state in trajectory])

    render_kwargs = {"width": width, "height": height}
    if camera is not None:
        render_kwargs["camera"] = camera

    last_error = None
    for candidate in trajectory_candidates:
        try:
            frames = env.render(candidate, **render_kwargs)
            return _normalize_frames(frames)
        except TypeError as exc:
            last_error = exc
            if "camera" in render_kwargs:
                try:
                    frames = env.render(candidate, width=width, height=height)
                    return _normalize_frames(frames)
                except Exception as inner_exc:
                    last_error = inner_exc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to render trajectory via env.render(...): {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a saved MuJoCo Playground SRGHN solution artifact to an mp4 video."
    )
    parser.add_argument("--solution", required=True, help="Path to a saved solution artifact (.pkl).")
    parser.add_argument("--output", default=None, help="Output mp4 path. Defaults to <solution-stem>.mp4.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for environment reset.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frames per second.")
    parser.add_argument("--width", type=int, default=640, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Render height in pixels.")
    parser.add_argument("--camera", default=None, help="Optional camera name passed through to env.render.")
    parser.add_argument(
        "--episode-horizon",
        type=int,
        default=None,
        help="Optional override for rollout horizon. Defaults to the saved training config.",
    )
    args = parser.parse_args()

    artifact = load_solution_artifact(args.solution)
    config = artifact["config"]
    if args.episode_horizon is not None:
        config = config.__class__(**{**config.__dict__, "episode_horizon": args.episode_horizon})

    env, trajectory, total_reward = _rollout_trajectory(
        artifact["individual"],
        config,
        key=jax.random.key(args.seed),
        obs_norm_state=artifact.get("obs_norm_state"),
    )
    frames = _render_frames(env, trajectory, width=args.width, height=args.height, camera=args.camera)

    output_path = Path(args.output) if args.output is not None else Path(args.solution).with_suffix(".mp4")
    _write_video_ffmpeg(frames, output_path, args.fps)
    print(
        f"Wrote {output_path} with {frames.shape[0]} frames at {args.fps} fps. "
        f"Episode return: {total_reward:.3f}"
    )


if __name__ == "__main__":
    main()
