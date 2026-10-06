# EigenDExplore

A fixed Eigen basis with one learnable log standard deviation per direction and policy, added to a policy's existing independent Gaussian noise.

```bash
python -m pip install ./packages/eigendexplore
```

Python 3.8+ and PyTorch 1.13+; no Action-Bench, simulator, or dataset dependencies.

## Use an existing PCA

```python
from eigendexplore import EigenDExplore

noise = EigenDExplore.from_pca(
    components,
    component_std,
    rank=10,
    coordinate_scale=joint_half_ranges,
    eigen_rms=0.5**0.5,
    spectrum_power=1.0,
)
```

These values, with an initial IID σ of √0.5, perform well across settings.

`components` has shape `(available_rank, action_dim)` with directions in rows; `component_std` contains their coefficient standard deviations, not variances. Columns must already follow policy action order, including the correct physical coordinate signs. All tensors share device and float32/float64 dtype. `coordinate_scale` is physical units per action unit; omit it if the components already use action coordinates. Defaults preserve the supplied PCA covariance without RMS reshaping; PCA means are not used for zero-mean exploration.

To use Action-Bench's packaged hand fits, install the core and call `from action_bench import load_eigendexplore; noise = load_eigendexplore("allegro")`. The [main README](../../README.md#use-it-in-your-own-environment) covers mirroring, action order, and extra wrist channels. Runnable [PPO](../../examples/ppo.py) and [sampling](../../examples/sampling.py) examples need no simulator.

## Supply a covariance

```python
import torch
from eigendexplore import EigenDExplore

# Replace these demonstration samples with your own N x action_dim matrix.
postures = torch.randn(1000, 16)
covariance = torch.cov(postures.T)
noise = EigenDExplore.from_covariance(covariance, rank=10, eigen_rms=0.5**0.5)
optimizer = torch.optim.Adam(noise.parameters(), lr=3e-4)

# Your policy supplies its usual mean and positive diagonal standard deviation.
mean = torch.zeros(64, 16)
iid_sigma = torch.full((16,), 0.5**0.5)
distribution = noise.distribution(mean, iid_sigma)
actions = distribution.sample()
log_prob = distribution.log_prob(actions)  # one joint-action log probability
entropy = distribution.entropy()

# Add noise.parameters() to your policy optimizer. In PPO, use this same
# distribution for rollout log probabilities and the training likelihood ratio.
loss = -log_prob.mean() - 0.01 * entropy.mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The optimizer above illustrates gradient flow; a real policy update also needs your algorithm's advantage estimates and objective. Save `noise.state_dict()` together with your policy checkpoint.

Covariance uses the policy's action coordinates. If you computed it in physical joint units, pass `coordinate_scale=joint_half_ranges` to convert normalized joint actions to those units. Columns must have the exact same order as the policy actions. Set `rank` explicitly to truncate; omit it to retain positive eigenvalues above numerical roundoff. With no shaping or normalization arguments, the added covariance reconstructs the supplied positive-semidefinite matrix to numerical precision.

`spectrum_power` raises component **standard deviations** to that power (default 1). `eigen_rms` optionally sets initial average per-action RMS of the correlated contribution. These are construction operations. Move the module once with `.to(device=..., dtype=...)`; rollout inputs must already match. Float32 and float64 are supported. Construction rejects malformed, nonsymmetric, or indefinite covariance matrices and rank-deficient bases. A zero covariance has no Eigen direction and is rejected.

You can instead supply `EigenDExplore(basis, eigen_sigma)` directly, with basis shape `(rank, action_dim)` and positive initial standard deviations shape `(rank,)`. Independent policies can supply `(policy_count, rank)` scales; select their rows explicitly through `distribution(mean, iid_sigma, eigen_sigma=selected_scales)`, matching the mean's batch dimensions. Directions need full row rank; they need not be orthonormal.

## Probability model

For independent standard normal vectors `epsilon` and `eta`:

```text
a = mu + iid_sigma * epsilon + B.T @ (exp(log_scale) * eta)
Cov[a] = diag(iid_sigma**2) + B.T @ diag(exp(log_scale)**2) @ B
```

The ordinary policy mean and IID noise remain. There is no Eigen mean head. PyTorch's low-rank multivariate Gaussian supplies sampling, exact action-space log probability, and entropy. The basis is a checkpointed buffer; `log_scale` is learnable. The policy can continue learning its IID scales independently.

## Correlated Eigen noise

Pass `correlation=rho` (in `[0, 1)`) to any constructor to make `eta` persist across steps:

```text
eta_t = rho * eta_{t-1} + sqrt(1 - rho**2) * xi_t,   xi_t ~ N(0, I)
```

Each step keeps the marginal above, and a sampled synergy lasts about `-1 / ln(rho)` steps. This suits held synergies such as grasps and chunked policies. The paper's experiments use the default, 0. Given `eta_{t-1}`, the action is Gaussian with mean `mu + rho * B.T @ (exp(log_scale) * eta_{t-1})` and Eigen scales `sqrt(1 - rho**2) * exp(log_scale)`. That conditional is the likelihood PPO needs:

```python
noise = EigenDExplore.from_pca(components, component_std, correlation=0.94)
state = noise.initial_state(mean.shape[:-1])  # stationary; redraw at resets
actions, next_state = noise.sample(mean, iid_sigma, state)
log_prob = noise.distribution(mean, iid_sigma, state=state).log_prob(actions)
```

Store each step's conditioning `state` with the rollout and pass it to `distribution(..., state=)` and `policy_kl(..., state=)` during the update. Without `state`, `distribution` returns the per-step marginal. Use its entropy for the entropy bonus. With `correlation=0`, `state` has no effect and `sample` draws from the marginal. The rl_games integration supports only the default.

## rl_games

The optional `eigendexplore.rl_games` module requires your host's rl_games installation. Before constructing its Runner:

```python
from eigendexplore.rl_games import configure

configure(
    agent_config["params"],
    {
        "basis": noise.basis.tolist(),
        "eigen_sigma": noise.log_scale.detach().exp().tolist(),
        "iid_sigma": [0.5**0.5] * noise.basis.shape[1],
    },
)
```

This registers a network and model through rl_games' public registry. The host still builds its own mean/value network. Here `iid_sigma` is an initial multiplier on the host's standard deviation, whose log-scale initializer should be zero for the stated initialization. The integration supports fixed and observation-conditioned IID scales. Its returned `sigmas` describe the IID component; training log probabilities and entropy include the correlated component.

SimToolReal's special SAPG behavior is handled by its benchmark integration; it requires no extra adapter in a custom environment.

`EigenDExplore.policy_kl(mean, iid_sigma, eigen_sigma, reference_mean, reference_iid_sigma, reference_eigen_sigma, reduce=True)` computes exact KL(current || reference). The Action-Bench release patches retain reference Eigen scales alongside native mean/IID snapshots and call this method for scheduler diagnostics. `reduce=False` returns per-sample values for recurrent masks. Standalone registry users must integrate those snapshots into their own trainer; returned `eigen_sigmas` alone does not change a trainer's KL calculation. The Gaussian and KL arithmetic run in the module's dtype while the policy network can use AMP.
