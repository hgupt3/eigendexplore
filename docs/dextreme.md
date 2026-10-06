# DeXtreme

Allegro in-hand cube reorientation (Isaac Gym, Python 3.8). Run the shared setup in the [README](../README.md#run-the-benchmarks) first.

## Install

Download **Isaac Gym Preview 4** from [NVIDIA](https://developer.nvidia.com/isaac-gym) and extract it outside this repository.

```bash
"$AB_CORE" scripts/setup_benchmark.py dextreme --root "$AB_WORK/dextreme" --prepare-only
conda create -y -p "$AB_WORK/env-dextreme" python=3.8 pip
AB_PY="$AB_WORK/env-dextreme/bin/python"
AB_GYM=/path/to/extracted/isaacgym
"$AB_PY" -m pip install --upgrade pip
"$AB_PY" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
"$AB_PY" -m pip install numpy==1.23.5
"$AB_PY" -m pip install -e "$AB_GYM/python"
"$AB_PY" -m pip install rl-games==1.6.5
"$AB_PY" -m pip install -e "$AB_WORK/dextreme/host"
"$AB_PY" -m pip install numpy==1.23.5 torch==2.4.1 ninja
"$AB_CORE" scripts/setup_benchmark.py dextreme --root "$AB_WORK/dextreme" --python "$AB_PY"
```

This writes `$AB_WORK/dextreme/installation.json`.

## Run

```bash
action-bench config --benchmark dextreme --hand allegro --method eigendexplore > study.yaml
action-bench run study.yaml --installation "$AB_WORK/dextreme/installation.json" \
  --output runs/dextreme/eigendexplore/seed-101 --seed 101 --gpu 0
```

Use any [method](../README.md#methods) in place of `eigendexplore`. Training output goes to the terminal and `train.log` in the output directory.

| Option | Effect |
| --- | --- |
| `--seed N` | Trial seed (default 101) |
| `--max-epochs N` | Shorter run, e.g. `--max-epochs 2` for a quick check |
| `--num-envs N` | Environments (default 16384, multiple of 1024) |
| `--prepare-only` | Write the resolved command and exported policy bundle without training |
| `--record` | Save videos ([README](../README.md#videos)) |
| `--checkpoint PATH` | Continue a previous run of the same study and seed into a new output directory |
