from __future__ import annotations

import os
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env, map_action_for_switch
from policy import apply_policy
from srghn import make_policy


def _to_uint8_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(jax.device_get(frame))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame ndim=3, got shape={arr.shape}.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected frame channels in {{1,3,4}}, got shape={arr.shape}.")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and np.nanmax(arr) <= 1.0 and np.nanmin(arr) >= 0.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_render_output(frames: Any) -> list[np.ndarray]:
    # Common structured render payloads: {"rgb": ...} or {"frames": ...}
    if isinstance(frames, dict):
        for key in ("rgb", "frames", "images", "image"):
            if key in frames:
                return _normalize_render_output(frames[key])
        raise ValueError(f"Unsupported render dict keys: {tuple(frames.keys())}")

    arr = np.asarray(jax.device_get(frames))
    if arr.ndim == 4:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 3:
        return [arr]
    if isinstance(frames, (list, tuple)):
        out = []
        for frame in frames:
            out.extend(_normalize_render_output(frame))
        return out
    raise ValueError("Unsupported render output format; expected array or sequence of arrays.")


def _stack_trajectory_states(pipeline_states: Sequence[Any]) -> Any:
    if not pipeline_states:
        raise ValueError("No pipeline states available for rendering.")
    if len(pipeline_states) == 1:
        return pipeline_states[0]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pipeline_states)


def _render_single_state_frames(render_fn, pipeline_states: Sequence[Any], kwargs: dict[str, Any]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for state in pipeline_states:
        out = render_fn(state, **kwargs)
        normalized = _normalize_render_output(out)
        if not normalized:
            continue
        # If renderer returned a short clip for one state, keep the first frame.
        frames.append(normalized[0])
    return frames


def _render_mujoco_playground_frames(
    env,
    pipeline_states: Sequence[Any],
    *,
    camera: str | None,
    width: int,
    height: int,
) -> list[np.ndarray]:
    render_fn = getattr(env, "render", None)
    if render_fn is None:
        raise ValueError("Environment does not expose a `render` method.")

    kwargs = {"width": width, "height": height}
    if camera is not None:
        kwargs["camera"] = camera

    traj = _stack_trajectory_states(pipeline_states)
    attempts = [
        lambda: render_fn(traj, **kwargs),
        lambda: render_fn(traj),
        lambda: render_fn(trajectory=traj, **kwargs),
        lambda: render_fn(states=traj, **kwargs),
        partial(_render_single_state_frames, render_fn, pipeline_states, kwargs),
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            frames = attempt()
            normalized = _normalize_render_output(frames)
            if normalized:
                return normalized
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    raise ValueError(
        "Failed to render MuJoCo Playground rollout. Tried multiple render signatures. "
        f"Errors: {' | '.join(errors)}"
    )


def _ensure_even_hw(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _save_mp4(frames: Sequence[np.ndarray], path: str, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to save.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "MP4 writing requires `imageio` and `imageio-ffmpeg`. "
            "Install them with `pip install imageio imageio-ffmpeg`."
        ) from exc

    frames_u8 = [_ensure_even_hw(_to_uint8_frame(frame)) for frame in frames]
    writer = imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=max(int(fps), 1),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for frame in frames_u8:
            writer.append_data(frame)
    finally:
        writer.close()


def record_mujoco_playground_rollout(
    srghn,
    config,
    output_path: str,
    *,
    key: jax.random.KeyArray,
    gen: int,
    render_every: int = 1,
    fps: int = 30,
    camera: str | None = "track",
    width: int = 640,
    height: int = 480,
) -> dict[str, float | int | str]:
    if render_every <= 0:
        raise ValueError("render_every must be positive.")

    env, _, _, _, is_discrete, action_shape, action_low, action_high = make_env(config)
    policy_params = make_policy(srghn)

    state = env.reset(key)
    obs = state.obs
    total_reward = 0.0
    pipeline_states = []
    steps = 0

    gen_arr = jnp.asarray(gen, dtype=jnp.int32)
    for t in range(config.episode_horizon):
        if t % render_every == 0:
            pipeline_states.append(getattr(state, "pipeline_state", state))

        action = apply_policy(policy_params, obs, is_discrete=is_discrete)
        action = map_action_for_switch(action, gen_arr, config)
        if is_discrete:
            action = jnp.asarray(action, dtype=jnp.int32)
        else:
            action = action.reshape(action_shape)
            if action_low is not None:
                action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)

        state = env.step(state, action)
        obs = state.obs
        total_reward += float(jnp.asarray(state.reward, dtype=jnp.float32))
        steps = t + 1
        done = bool(jnp.asarray(state.done, dtype=jnp.bool_))
        if done:
            break

    if steps == 0 or steps % render_every != 0:
        pipeline_states.append(getattr(state, "pipeline_state", state))

    frames = _render_mujoco_playground_frames(
        env,
        pipeline_states,
        camera=camera,
        width=width,
        height=height,
    )
    _save_mp4(frames, output_path, fps=fps)

    return {
        "path": output_path,
        "episode_return": total_reward,
        "steps": steps,
        "frames": len(frames),
        "fps": fps,
    }
