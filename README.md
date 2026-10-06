# EigenDExplore

**[Project page](https://eigendexplore.github.io)**

EigenDExplore adds exploration noise along human hand-motion eigenvectors to a policy's ordinary joint-space noise. The policy still outputs joint targets. This repository contains the method, a fitted EgoSuite basis for six robot hands, and the four benchmarks from the paper.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install ./packages/eigendexplore .
action-bench list
```

## Use it in your own environment

```python
from action_bench import load_eigendexplore

noise = load_eigendexplore("allegro")  # side="left" for a left hand
distribution = noise.distribution(mean, iid_sigma)  # replaces your diagonal Gaussian
actions = distribution.sample()
log_prob = distribution.log_prob(actions)
```

- Start your policy's IID σ at √0.5. With the loader's defaults (Eigen RMS √0.5, spectrum power 1), this performs well across settings.
- Make `noise` a submodule of your policy so its learned scales are optimized and checkpointed.
- Use this distribution for log-probabilities, entropy, and KL (`noise.policy_kl(...)`). Store the raw, unclamped sampled actions.
- Actions are absolute joint targets normalized to [-1, 1], in `load_hand_layout("allegro_right").joint_names` order. For another order or extra wrist/arm channels, pass `action_joint_names=`. For other units, pass `coordinate_scale=` (physical units per action unit, one entry per action).
- Packaged hands: Allegro, Inspire, Schunk, Sharpa, Wuji, XHand. For another hand, install only `./packages/eigendexplore` and use your own PCA: `EigenDExplore.from_pca(components, component_std, rank=k, coordinate_scale=...)`.

### Correlated Eigen noise

By default each step draws fresh Eigen noise, so a sampled synergy lasts one step. When a synergy must be held, as in closing and keeping a grasp or when the policy acts in chunks, let it persist across steps. This variant is not in the paper. Initial testing shows it works really well, potentially better than the default:

```python
noise = load_eigendexplore("allegro", correlation=0.94)  # a synergy lasts ~16 steps
state = noise.initial_state(mean.shape[:-1])  # redraw at episode resets

actions, next_state = noise.sample(mean, iid_sigma, state)
log_prob = noise.distribution(mean, iid_sigma, state=state).log_prob(actions)
state = next_state
```

- The Eigen coefficients follow `η_t = ρ η_{t−1} + √(1−ρ²) ξ_t`. Each step keeps the default spread, and a synergy persists about `−1/ln ρ` steps. Match that to one action chunk. IID noise stays independent.
- Store each step's `state` (the one it was sampled from) with the rollout. Pass it as `state=` for PPO log-probabilities and `noise.policy_kl(...)`. Use `noise.distribution(mean, iid_sigma).entropy()` for the entropy bonus.
- `correlation=0` is the paper's EigenDExplore. The benchmarks and the rl_games integration use it.

CPU examples:

```bash
python examples/ppo.py --updates 4
python examples/ppo.py --updates 4 --correlation 0.94
python examples/sampling.py
```

## Run the benchmarks

| Benchmark | Hands | Guide |
| --- | --- | --- |
| DeXtreme | Allegro | [docs/dextreme.md](docs/dextreme.md) |
| SimToolReal | Sharpa | [docs/simtoolreal.md](docs/simtoolreal.md) |
| SPIDER | Sharpa, Wuji, Allegro, XHand, Inspire, Schunk | [docs/spider.md](docs/spider.md) |

Each benchmark installs into its own simulator environment. Start with this shared setup on a Linux machine with an NVIDIA GPU and [Conda](https://github.com/conda-forge/miniforge#install):

```bash
AB_SOURCE="$PWD"                      # this checkout
AB_WORK=/path/to/local/disk/action-bench
mkdir -p "$AB_WORK"
conda create -y -p "$AB_WORK/core" python=3.10 pip
AB_CORE="$AB_WORK/core/bin/python"
"$AB_CORE" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$AB_CORE" -m pip install -e "$AB_SOURCE/packages/eigendexplore" -e "$AB_SOURCE"
export PATH="$AB_WORK/core/bin:$PATH"
```

An RL run trains one seed. To run more seeds, repeat the command with a different `--seed` and a new `--output`.

### Methods

| Method | Paper name | Action |
| --- | --- | --- |
| `joint_absolute` | Joint-Space | Absolute joint targets |
| `eigendexplore` | EigenDExplore | Absolute joint targets, IID + Eigen noise |
| `eigen_absolute` | Eigen-Space | Absolute Eigen posture (k actions) |
| `joint_absolute_eigen_residual` | Eigen-Residual | Joint targets + 0.3 × centered Eigen posture |
| `joint_delta` | Joint-Δ (appendix) | Integrated joint increments, 3 ranges/s |
| `eigen_delta_joint_delta` | Eigen-Residual-Δ (appendix) | Integrated Eigen and joint increments, 3/s each |

DeXtreme and SimToolReal run all six. Defaults for each benchmark are in [`src/action_bench/presets/`](src/action_bench/presets); change one with `action-bench config ... --set name=value`.

### Videos

Add `--record` to an RL run to save scheduled 30-second state captures and render them to MP4. It needs the renderer in the core environment (Ubuntu also needs `libosmesa6`):

```bash
"$AB_CORE" -m pip install -e "$AB_SOURCE[recording]"
"$AB_CORE" -m pip install --no-deps PyOpenGL==3.1.10
```

Videos appear as `recordings/step_<epoch>.mp4` in the run directory. Re-render a capture with `action-bench render <capture-dir> --installation <installation.json> --output out.mp4`. For SPIDER, `--record` saves each episode's video.

## Repository layout

| Path | Contents |
| --- | --- |
| `packages/eigendexplore/` | Standalone PyTorch distribution and rl_games integration |
| `src/action_bench/` | Basis loading, hand layouts, action maps, CLI, benchmark adapters |
| `src/action_bench/assets/` | Fitted EgoSuite basis per hand |
| `src/action_bench/presets/` | Default settings per benchmark |
| `taskpacks/` | Task inputs for SPIDER and DeXtreme |
| `third_party/`, `scripts/setup_benchmark.py` | Pinned benchmark sources, patches, installer |

`AGENTS.md` holds project conventions for collaborators and coding agents. Machine-specific notes go in a gitignored `AGENTS.local.md`.
