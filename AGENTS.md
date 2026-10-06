# Action-Bench agent instructions

Action-Bench compares hand-policy methods in DeXtreme and SimToolReal, plus cold-start optimizer sampling in Spider. This repository prefers coherent replacement: no legacy aliases, migration shims, fuzzy matching, or retired implementations in the source tree. Retired work remains in git history.

## Shared and local instructions

This file holds the shared project instructions for every collaborator and coding agent. It must not depend on one person's home directory, machines, or private infrastructure.

Machine-specific notes go in a gitignored `AGENTS.local.md` beside this file (or a symlink to a private file): storage roots, interpreter paths, GPU etiquette, runs, and installation records. At session start, agents read `AGENTS.local.md` when it exists, plus any files it says to read. Never put machine-specific absolute paths in tracked files. Benchmark commands receive paths through CLI arguments and installation records; runtime code does not read these notes.

## Public scope

- DeXtreme and SimToolReal expose exactly six methods, the paper's arms: `joint_absolute` (Joint-Space), `eigendexplore`, `eigen_absolute` (Eigen-Space), `joint_absolute_eigen_residual` (Eigen-Residual), and the appendix `joint_delta` and `eigen_delta_joint_delta`. Variants that no paper result uses are not shipped as benchmark methods. The one exception is the standalone package's opt-in correlated Eigen noise (`correlation`, default 0), offered for custom environments only.
- Public fitted representations are called **Eigen bases**. Internal PCA classes and original manifest fields describe their mathematical format; they are not additional user-facing methods.
- Ship one fitted EgoSuite basis per available hand (`assets/{hand}/`), with its fit metadata. Do not ship raw data, representation-training corpora, or a producer workflow. Fixed benchmark reference inputs are included under `taskpacks/`; they define the tasks and are not representation-training datasets. There is no dataset selector.
- Layout and artifact availability are data. Benchmark presets explicitly constrain what their pinned hosts support.
- Spider compares IID and EigenDExplore sampling for one hand per run: ten tasks, ten paired seeds, one shared IID baseline. Defaults are the human basis, 90% retained variance, and IID/Eigen scales 1/1; `--set` sweeps basis (`human`/`random`), retained variance, and scales (IID scale 0 is Eigen noise alone). Both methods start from the hand's fixed cold-start posture in `spider/configs/hands.yaml`. The task archive is tracked under `taskpacks/`, outside the Python wheel.
- Spider keeps the original power-one eigenvalue spectrum, normalized to the hand's fixed `eigen_noise_rms` before applying scales. It uses native optimizer selection and exact host coupling; PPO log-scale learning is not involved.

## Architecture

`packages/eigendexplore` is the standalone PyTorch distribution and rl_games registry integration. `src/action_bench` provides the benchmark-independent `load_eigendexplore` API and owns hand layouts, five batched target transformations, the packaged basis catalog, strict study settings, and thin host adapters. Hosts own wrists, arms, objects, rewards, physics, training, and native mimic expansion. Ordered joint names are the boundary.

EigenDExplore changes exploration, not the target map. Actions have mean mu and covariance `diag(iid_sigma**2) + B.T @ diag(exp(log_scale)**2) @ B`. IID noise remains; each fixed eigen direction gets its own learnable log scale and no mean head. Sampling, log probability, and entropy all use this same action-space marginal. rl_games uses its public model/network registry, with no new sampler fork. Both RL hosts use pinned rl_games sources with release patches that preserve covariance references and compute exact EigenDExplore KL. SimToolReal gives each SAPG group independent Eigen scales and relabels critic inputs for shared experience. Checkpoint restore synchronizes scheduler and optimizer learning rates.

With `correlation` rho > 0 the Eigen coefficients follow an AR(1) process with the same per-step marginal. Likelihoods and KL condition on the stored previous state. Entropy stays on the marginal. The benchmarks and rl_games integration keep rho = 0.

Defaults belong in `presets/{benchmark}.yaml`, not named variants. EigenDExplore initializes `(iid_sigma, eigen_rms, spectrum_power)` to `(1, 1, 0.5)` for SimToolReal and `(sqrt(0.5), sqrt(0.5), 1)` for DeXtreme. These are initial conditions; training learns scales.

## Optional recording

The recording workflow is retained intentionally. `--record` enables scheduled replayable state capture and automatic local video rendering for the RL benchmarks; Spider uses its native episode video. Recording is disabled by default, imports renderer dependencies only when requested, and uploads nothing automatically. Preserve the four-view SimToolReal grid, reset alignment, and fixed epoch schedules. Shared capture modules run standalone in DeXtreme's Python 3.8 host; CPU rendering runs in the modern launcher environment. See the README and `docs/{benchmark}.md` for installation and replay.

## Runtime contracts

- One artifact per hand family, serving both sides through the signed canonical-right layout permutation.
- Structural errors fail: schema, dimensions, joint order, nonfinite construction inputs, incompatible device/dtype. Quality measurements remain statistics, not acceptance gates.
- No artifact/layout checksums. Original artifact IDs, recorded fit configurations, exact joint order, release versions, and pinned host commits identify inputs.
- Tensor operations during rollout are batched; no environment/joint loops, NumPy conversions, parsing, automatic device transfers, or host synchronization. Validate fixed data once during construction.
- Eigen action maps observe measured `encode(qpos)`. Integrating Eigen maps also observe `encode(commanded target)`. Joint integrators observe the commanded target where the host lacks it (DeXtreme). Order is measured latent, commanded joint target, commanded latent. Absolute maps add no commanded state. EigenDExplore adds no representation observations.
- Delta maps integrate in joint space, rebasing on the host's filtered commanded target. No latent accumulator survives in this slate.
- Joint-absolute plus Eigen residual is memoryless: `clamp(Q(a_joint) + fraction * (decode(z) - decode(0)))`. Zero fraction recovers joint absolute.
- Rates specify delivered authority per second. Raw delta scale is `rate / (target_ema * control_hz)`. Current registered rates are 3.0 joint ranges/s and 3.0 Eigen coefficient spans/s. Eigen residual fraction is 0.3. DeXtreme uses stock ADR EMA annealing, calibrated at 0.175 and 30 Hz; SimToolReal uses 0.1 and 60 Hz.

## Reproducibility

`third_party/benchmarks.lock.yaml` pins host and training sources. Setup patches those exact sources in a new installation directory. Each run records resolved settings, geometry, seed, host commit, artifact identity, command, and environment. Resume requires a matching recorded release run; retired runs cannot silently resume under renamed settings. A trial uses an explicit seed and fresh output directory: RL defaults to 101, Spider to its ten-seed list. Accept only seeds 0 through 4294967295; no implicit random-seed sentinel.

DeXtreme's Python 3.8 host loads a tensor-only TorchScript export made in the modern core environment. The standalone EigenDExplore package works across all three host Python versions. Keep simulators in separate environments. Agents setting up a fresh machine must follow `docs/{benchmark}.md`: prepare pinned sources, install the selected simulator dependencies, finish package registration, then validate a smoke run. Do not treat a successful source checkout or import probe as a completed simulator test.

Custom-environment examples run on CPU and use packaged fits without benchmark Study settings. Standalone external-PCA use needs no catalog or hand registration.

The release ships no test suite. Verify changes by running the CPU examples and a short benchmark run. Raw data, runs, and checkpoints remain external and are never declared disposable from their names alone.
