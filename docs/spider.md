# SPIDER

Sampling-based trajectory optimization on ten OakInk2 object-tracking tasks (MuJoCo Warp, Python 3.12). Each run compares IID joint noise with EigenDExplore noise on one hand, paired by task and seed, and reports the final retargeting cost (lower is better). Run the shared setup in the [README](../README.md#run-the-benchmarks) first.

## Install

You need a C/C++ compiler and Python development headers (Conda Python includes them).

```bash
"$AB_CORE" scripts/setup_benchmark.py spider --root "$AB_WORK/spider" --prepare-only
conda create -y -p "$AB_WORK/env-spider" python=3.12 pip
AB_PY="$AB_WORK/env-spider/bin/python"
"$AB_PY" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$AB_PY" -m pip install mujoco==3.7.0 mujoco-warp==3.7.0.1 warp-lang==1.12.1 \
  numpy loguru hydra-core 'imageio[ffmpeg]' opencv-python \
  rerun-sdk==0.26.2 viser trimesh tyro pandas scipy matplotlib
"$AB_PY" -m pip install -e "$AB_SOURCE/packages/eigendexplore" -e "$AB_SOURCE"
"$AB_CORE" scripts/setup_benchmark.py spider --root "$AB_WORK/spider" --python "$AB_PY"
mkdir -p "$AB_WORK/data"
tar -xzf "$AB_SOURCE/taskpacks/spider-oakinkv2-right-10task-v3.tar.gz" -C "$AB_WORK/data"
```

## Run

```bash
action-bench spider run --hand sharpa \
  --installation "$AB_WORK/spider/installation.json" \
  --assets "$AB_WORK/data/spider-oakinkv2-right-10task-v3" \
  --output runs/spider/sharpa --gpu 0
action-bench spider report runs/spider/sharpa
```

This runs all 10 tasks × 10 seeds for IID and EigenDExplore (200 episodes) and writes `comparison.csv` / `comparison.json`. Hands: `sharpa`, `wuji`, `allegro`, `xhand`, `inspire`, `schunk`.

| Option | Effect |
| --- | --- |
| `--task NAME` | Run only this task (repeatable) |
| `--seed N` | Run only this seed (repeatable); default seeds 973500–973509 |
| `--prepare-only` | Write the episode plan without running |
| `--resume` | Continue an interrupted run with the same settings |
| `--record` | Save each episode's video |

For a quick check, add `--task pick_spoon_bowl --seed 973500` (2 episodes).

## Change the EigenDExplore settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `basis` | human | `human` EgoSuite directions, or `random` orthonormal directions with the same per-direction scales |
| `retained_variance` | 0.9 | Fraction of PCA variance kept; sets the number of directions k |
| `iid_scale` | 1.0 | Joint-space noise scale; 0 gives Eigen noise only |
| `eigen_scale` | 1.0 | Eigen noise scale |

Set them with `--set`. A comma-separated list sweeps the values, and every combination is compared against one shared IID baseline. The paper's retained-variance ablation (Fig. 6b) is:

```bash
action-bench spider run --hand sharpa \
  --set retained_variance=0.5,0.7,0.8,0.9,0.95,1.0 --set iid_scale=0,1 \
  --installation "$AB_WORK/spider/installation.json" \
  --assets "$AB_WORK/data/spider-oakinkv2-right-10task-v3" \
  --output runs/spider/sharpa-ablation --gpu 0
```

For the human-versus-random directions comparison (App. B.2), replace the two `--set` flags with:

```bash
  --set basis=human,random --set iid_scale=0,1
```

The report has one row per setting.
