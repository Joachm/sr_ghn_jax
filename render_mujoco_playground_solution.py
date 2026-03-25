from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from envs import make_env, map_action_for_switch
from obs_norm import normalize_obs
from policy import apply_policy
from solution_artifacts import load_solution_artifact
from srghn import make_policy


def _available_camera_names(env) -> list[str]:
    model = getattr(env, "mj_model", None) or getattr(env, "_mj_model", None) or getattr(env, "model", None)
    if model is None or not hasattr(model, "ncam") or not hasattr(model, "name_camadr") or not hasattr(model, "names"):
        return []

    names: list[str] = []
    for i in range(int(model.ncam)):
        start = int(model.name_camadr[i])
        raw_name = model.names[start:]
        if isinstance(raw_name, memoryview):
            raw_name = raw_name.tobytes()
        if isinstance(raw_name, str):
            name = raw_name.split("\x00", 1)[0]
        else:
            name = bytes(raw_name).split(b"\x00", 1)[0].decode("utf-8", "ignore")
        names.append(name)
    return names


def _choose_camera(env, requested: str | None) -> str | None:
    camera_names = _available_camera_names(env)
    if requested is not None:
        return requested
    if not camera_names:
        return None

    preferred = ("track", "tracking", "follow", "close", "side", "run")
    lowered = {name.lower(): name for name in camera_names}
    for token in preferred:
        for lowered_name, original_name in lowered.items():
            if token in lowered_name:
                print(f"[render] auto-selected camera {original_name!r}")
                return original_name

    if camera_names and any(name for name in camera_names):
        print(f"[render] available cameras: {', '.join(repr(name) for name in camera_names)}")
    return None


def _body_names(model) -> list[str]:
    if not hasattr(model, "nbody") or not hasattr(model, "name_bodyadr") or not hasattr(model, "names"):
        return []
    names: list[str] = []
    for i in range(int(model.nbody)):
        start = int(model.name_bodyadr[i])
        raw_name = model.names[start:]
        if isinstance(raw_name, memoryview):
            raw_name = raw_name.tobytes()
        if isinstance(raw_name, str):
            name = raw_name.split("\x00", 1)[0]
        else:
            name = bytes(raw_name).split(b"\x00", 1)[0].decode("utf-8", "ignore")
        names.append(name)
    return names


def _default_track_body_id(model) -> int:
    body_names = _body_names(model)
    if not body_names:
        return -1
    preferred = ("torso", "trunk", "pelvis", "body", "root", "base")
    for token in preferred:
        for idx, name in enumerate(body_names):
            if idx == 0:
                continue
            if token in name.lower():
                return idx
    return 1 if len(body_names) > 1 else -1


def _should_use_tracking_camera(config, camera: str | None) -> bool:
    if camera is not None:
        return False
    env_id = str(getattr(config, "env_id", "")).lower()
    return "cheetah" in env_id


def _tracking_camera(model) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = _default_track_body_id(model)
    cam.distance = max(float(model.stat.extent) * 1.5, 2.5)
    cam.azimuth = 90.0
    cam.elevation = -12.0
    cam.lookat[:] = 0.0
    return cam


def _state_to_mjdata(model, state):
    data = mujoco.MjData(model)
    state_data = getattr(state, "data", None)
    if state_data is None:
        raise ValueError("Wrapped environment state is missing `.data`, which is required for tracked rendering.")
    data.qpos[:] = np.asarray(jax.device_get(state_data.qpos))
    data.qvel[:] = np.asarray(jax.device_get(state_data.qvel))
    if hasattr(state_data, "act") and data.act.size:
        data.act[:] = np.asarray(jax.device_get(state_data.act))
    if hasattr(state_data, "mocap_pos") and data.mocap_pos.size:
        data.mocap_pos[:] = np.asarray(jax.device_get(state_data.mocap_pos))
    if hasattr(state_data, "mocap_quat") and data.mocap_quat.size:
        data.mocap_quat[:] = np.asarray(jax.device_get(state_data.mocap_quat))
    mujoco.mj_forward(model, data)
    return data


def _render_frames_tracking(env, trajectory, *, width: int, height: int) -> np.ndarray:
    model = getattr(env, "mj_model", None) or getattr(env, "_mj_model", None) or getattr(env, "model", None)
    if model is None:
        raise RuntimeError("Unable to access MuJoCo model for tracked rendering.")
    camera = _tracking_camera(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    for idx, state in enumerate(trajectory):
        data = _state_to_mjdata(model, state)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
        if (idx + 1) % 100 == 0:
            print(f"[render] rendered frame {idx + 1}/{len(trajectory)} (tracking camera)")
    renderer.close()
    return _normalize_frames(np.stack(frames, axis=0))


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

    def step_fn(state):
        obs = state.obs
        obs_in = normalize_obs(obs, obs_norm_state, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
        action = apply_policy(policy_params, obs_in, config, is_discrete=is_discrete)
        action = map_action_for_switch(action, gen, config)
        if is_discrete:
            action = jnp.asarray(action, dtype=jnp.int32)
        else:
            action = action.reshape(action_shape)
            if action_low is not None:
                action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
        return env.step(state, action)

    step_jit = jax.jit(step_fn)
    state = env.reset(key)
    trajectory = [jax.device_get(state)]
    total_reward = 0.0
    gen = jnp.asarray(config.num_generations - 1, dtype=jnp.int32)

    for step_idx in range(config.episode_horizon):
        state = step_jit(state)
        reward = float(jax.device_get(jnp.asarray(state.reward, dtype=jnp.float32)))
        done = bool(jax.device_get(jnp.asarray(state.done)))
        total_reward += reward
        trajectory.append(jax.device_get(state))
        if (step_idx + 1) % 100 == 0:
            print(f"[render] rollout step {step_idx + 1}/{config.episode_horizon}")
        if done:
            break

    return env, trajectory, total_reward


def _render_frames(env, trajectory, *, config, width: int, height: int, camera: str | None):
    if not hasattr(env, "render"):
        raise RuntimeError("Environment does not expose env.render(trajectory, ...).")

    if _should_use_tracking_camera(config, camera):
        try:
            return _render_frames_tracking(env, trajectory, width=width, height=height)
        except Exception as exc:
            print(f"[render] tracking camera fallback failed, falling back to env.render(...): {exc}")

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
            # Some MuJoCo Playground envs require the full wrapped state rather than
            # bare MJX pipeline/data objects; fall through and try the next candidate.
            continue

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
        "--list-cameras",
        action="store_true",
        help="List available camera names for the environment and exit.",
    )
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

    t0 = perf_counter()
    print("[render] rolling out trajectory...")
    env, trajectory, total_reward = _rollout_trajectory(
        artifact["individual"],
        config,
        key=jax.random.key(args.seed),
        obs_norm_state=artifact.get("obs_norm_state"),
    )
    camera_names = _available_camera_names(env)
    if camera_names:
        print(f"[render] available cameras: {', '.join(repr(name) for name in camera_names)}")
    if args.list_cameras:
        return
    camera = _choose_camera(env, args.camera)
    print(f"[render] rollout finished in {perf_counter() - t0:.1f}s with {len(trajectory)} states")
    t1 = perf_counter()
    print("[render] rendering frames...")
    frames = _render_frames(env, trajectory, config=config, width=args.width, height=args.height, camera=camera)
    print(f"[render] rendering finished in {perf_counter() - t1:.1f}s with {frames.shape[0]} frames")

    output_path = Path(args.output) if args.output is not None else Path(args.solution).with_suffix(".mp4")
    t2 = perf_counter()
    print(f"[render] encoding mp4 to {output_path}...")
    _write_video_ffmpeg(frames, output_path, args.fps)
    print(f"[render] encoding finished in {perf_counter() - t2:.1f}s")
    print(
        f"Wrote {output_path} with {frames.shape[0]} frames at {args.fps} fps. "
        f"Episode return: {total_reward:.3f}"
    )


if __name__ == "__main__":
    main()
