# Reference Spec (extracted)

This document captures the explicit constants and behavioral rules found in the
reference repo so the JAX reimplementation can match it without guessing.
Source files: `experiments.py`, `pybullet_experiments.py`, `utility.py`.

## Experiments and env names
- CartPole experiment: `ENV_NAME = 'CartPole-v1'` in `experiments.py` (lines 329-331).
- Ant (PyBullet) experiment: `ENV_NAME = 'AntPyBulletEnv-v0'` in
  `pybullet_experiments.py` (lines 329-331).
 

## Policy architecture per task
SimpleMLP is a feedforward MLP with `Tanh` activations after each hidden layer
and a final linear output layer (experiments.py:78-101,
pybullet_experiments.py:81-104).

- CartPole (CartPole-v1):
  - `MLP_ARCH = (4, [32], 2)` (experiments.py:337-338).
  - obs dim: 4, action dim: 2, hidden layers: [32], activation: `Tanh`.
- Ant (AntPyBulletEnv-v0):
  - `MLP_ARCH = (28, [32, 32, 32], 8)` (pybullet_experiments.py:335-337).
  - obs dim: 28, action dim: 8, hidden layers: [32, 32, 32], activation: `Tanh`.
 

Policy rollout details:
- `env_rollout` in `experiments.py` slices observations: `obs = obs[:27]`
  before passing into the policy (experiments.py:227-229). This applies to
  CartPole as written.
- `env_rollout` in `pybullet_experiments.py` does not slice observations
  (pybullet_experiments.py:297-312).

## Evolution settings per task
CartPole (`experiments.py`):
- `population_size = 10` (line 342).
- `generations = 300` (line 343).
- `n_children = 2` (line 344).
- `child_factor = 0.5` (line 345).
- `hidden_dim = 32` (line 341).
- `REPS = 1` (line 340).

Ant (PyBullet) (`pybullet_experiments.py`):
- `population_size = 50` (line 341).
- `generations = 1000` (line 342).
- `n_children = 2` (line 343).
- `child_factor = 0.5` (line 344).
- `hidden_dim = 32` (line 340).
- `REPS = 1` (line 339).

Elite selection:
- All candidates are `population + children`, then sorted by fitness ascending
  and the top `population_size` are kept (experiments.py:381-435,
  pybullet_experiments.py:380-435).

Parent fitness mixing:
- A parent's fitness is replaced by a weighted average of its own fitness and
  the mean of its children: `(1 - child_factor) * parent + child_factor * mean(children)`
  (experiments.py:420-428, pybullet_experiments.py:419-427).

## Switch rules
- Reference does not define switch rules; project owner clarified:
  - CartPole: after a configured generation threshold, actions are reversed.
    For discrete actions, this is `action = 1 - action` for `action_dim = 2`.
  - Switch window uses `switch_gen_start` and optional `switch_gen_end` from
    config; if `switch_gen_end` is `None`, the switch applies to all later
    generations.

## Mutation / update generation rules
Hypernetwork structure and update sampling (StochasticHyperNetwork):
- trunk: Linear(emb_dim -> 64, bias=False) + ReLU + Linear(64 -> 64, bias=False)
  + ReLU (experiments.py:48-54, pybullet_experiments.py:51-57).
- log-std head: Linear(64 -> coeff_dim, bias=False) (experiments.py:55,
  pybullet_experiments.py:58).
- lr head: Linear(64 -> 1, bias=True) then `sigmoid` (experiments.py:56, 66;
  pybullet_experiments.py:59, 69).
- std clamp: `std_c = clip(exp(log_std_c), 0., 2.)` (experiments.py:67,
  pybullet_experiments.py:70).
- sample (reference): `eps ~ Normal(0, std_c)` then `c = eps * std_c`
  (experiments.py:69-70, pybullet_experiments.py:72-73). Note: JAX port uses
  a single scaling `c = Normal(0, std_c)` to avoid sigma-squared variance.
- basis: `basis` is orthogonally initialized with shape
  `(coeff_dim, max_params)` (experiments.py:58-60, pybullet_experiments.py:61-63).
  - The basis is registered as a buffer in the reference (not a parameter), so
    it should not be mutated by the evolutionary update.
- update vector: `w_out = c @ basis`, then
  `clip(w_out, -1, 1) * lr + Normal(0, 0.001)` (experiments.py:71-72,
  pybullet_experiments.py:74-75).

Parameter update application (make_children):
- Each child clones the parent and adds the generated update to parameters
  (`p.add_(param_update[n])`) and then clamps parameters to `[-20, 20]`
  (experiments.py:288-295, pybullet_experiments.py:259-267).
- No additional update clamp is applied beyond the hypernetwork output clip
  and parameter clamp noted above.

## Mutation rate computation
- The mutation rate `lr` is a scalar produced by a `Linear(64 -> 1)` head with
  `sigmoid` (experiments.py:56, 66; pybullet_experiments.py:59, 69).
- The lr is applied as a multiplier on the clipped `w_out` (experiments.py:71-72,
  pybullet_experiments.py:74-75).
  - JAX implementation detail: when `mutation_rate_head_dim > 1`, use
    `max(sigmoid(lr_logits))` to reduce to a scalar. This matches the reference
    behavior when the head has output dim 1.

## Graph construction rules
Policy/control MLP graph:
- Built as an ordered, sequential graph from parameter shapes using
  `build_ordered_graph_from_shapes`: nodes in the order of `shapes.keys()`,
  edges between consecutive nodes (utility.py:43-50).

Self graph (meta GHN) construction:
- During initialization, a dummy GHN is created and its param shapes are
  extracted. An extra entry `embed.weight` is appended to `shapes_dummy` with
  shape `(len(shapes_dummy.keys()), hidden_dim)` (experiments.py:352-356,
  pybullet_experiments.py:351-355).
- A dummy ordered graph `G_dummy` is built from `shapes_dummy`
  (experiments.py:357, pybullet_experiments.py:356).
- A meta GHN is built with node types = `shapes_dummy.keys()` and
  `node_param_dims` = per-parameter size from `shapes_dummy` (experiments.py:359-363,
  pybullet_experiments.py:358-362).
- The meta GHN graph is built via `build_graph_with_hooks`, which uses forward
  hooks to connect parameter nodes based on tensor flow, and adds sequential
  edges within each module (utility.py:55-106). This hook-based graph builder
  is not available in JAX and must be replicated with a static rule.
  - JAX implementation compromise: use a bidirectional chain graph over
    self-parameter leaf indices (pytree leaf order). This preserves the ordered
    parameter-node structure without dynamic hooks.

## Notes / gaps
- No explicit switch-generation schedule or action remapping rules are defined
  in the reference files. If switching is required, the exact rules must be
  provided externally.
 
