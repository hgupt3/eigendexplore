# SimToolReal

Sharpa hand-arm tool manipulation (Isaac Sim / IsaacLab, Python 3.11). Run the shared setup in the [README](../README.md#run-the-benchmarks) first.

## Install

```bash
"$AB_CORE" scripts/setup_benchmark.py simtoolreal --root "$AB_WORK/simtoolreal" --prepare-only
conda create -y -p "$AB_WORK/env-simtoolreal" python=3.11 pip
AB_PY="$AB_WORK/env-simtoolreal/bin/python"
"$AB_PY" -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
"$AB_PY" -m pip install 'isaaclab[isaacsim,all]==2.3.2.post1' \
  --extra-index-url https://pypi.nvidia.com
"$AB_PY" -m pip install omegaconf hydra-core gym==0.23.1 scipy numpy==1.26.0 \
  yourdfpy requests tqdm tyro 'imageio[ffmpeg]' wandb termcolor coacd \
  'typing_extensions>=4.13' tensorboard tensorboardX psutil setproctitle
"$AB_PY" -m pip install -e "$AB_SOURCE/packages/eigendexplore" -e "$AB_SOURCE"
"$AB_CORE" scripts/setup_benchmark.py simtoolreal --root "$AB_WORK/simtoolreal" --python "$AB_PY"
export OMNI_KIT_ACCEPT_EULA=YES   # after accepting NVIDIA's Isaac Sim terms
```

This writes `$AB_WORK/simtoolreal/installation.json`.

## Run

```bash
action-bench config --benchmark simtoolreal --hand sharpa --method eigendexplore > study.yaml
action-bench run study.yaml --installation "$AB_WORK/simtoolreal/installation.json" \
  --output runs/simtoolreal/eigendexplore/seed-101 --seed 101 --gpu 0
```

Use any [method](../README.md#methods) in place of `eigendexplore`. The success tolerance stays at 6 cm / 50° for the first 9B frames, then tightens toward 1 cm / 8° as the policy improves. Training output goes to the terminal and `train.log` in the output directory. Set `WANDB_MODE=offline` to log without a W&B account.

| Option | Effect |
| --- | --- |
| `--seed N` | Trial seed (default 101) |
| `--max-epochs N` | Shorter run, e.g. `--max-epochs 2` for a quick check |
| `--num-envs N` | Environments (default 12288, multiple of 12288) |
| `--prepare-only` | Write the resolved command without training |
| `--record` | Save videos ([README](../README.md#videos)) |
| `--checkpoint PATH` | Continue a previous run of the same study and seed into a new output directory |
