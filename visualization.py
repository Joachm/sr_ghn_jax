from __future__ import annotations

import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

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

    attempts = [
        lambda: render_fn(pipeline_states, **kwargs),
        lambda: render_fn(pipeline_states),
        lambda: render_fn(trajectory=pipeline_states, **kwargs),
        lambda: render_fn(states=pipeline_states, **kwargs),
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


def _save_gif(frames: Sequence[np.ndarray], path: str, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to save.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_frames = [Image.fromarray(_to_uint8_frame(frame)) for frame in frames]
    duration_ms = max(1, int(round(1000.0 / max(fps, 1))))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


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
    _save_gif(frames, output_path, fps=fps)

    return {
        "path": output_path,
        "episode_return": total_reward,
        "steps": steps,
        "frames": len(frames),
        "fps": fps,
    }
