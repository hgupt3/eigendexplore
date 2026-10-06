# Host integrations

`benchmarks.lock.yaml` pins external source revisions. Patches apply only to those revisions and are installed into separate host environments. Host software and assets retain their upstream licenses.

Spider comes from `facebookresearch/spider`; its upstream `LICENSE` is headed **CC-by-NC License**, copyright Meta Platforms, Inc. and affiliates. The Spider patch modifies its runtime and optimizer integration and is distributed subject to that upstream license. The original copyright notices remain in the patched sources. This repository's MIT software license does not relicense Spider or benchmark/dataset assets.

The software license files at the selected revisions identify:

| Source | Software license | Selected source |
| --- | --- | --- |
| IsaacGymEnvs | BSD 3-Clause | [LICENSE.txt](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/LICENSE.txt) |
| SimToolReal | MIT | [LICENSE](https://github.com/hgupt3/simtoolreal/blob/fddff7462bfb683becb8aa0d73d496dca6065af4/LICENSE) |
| rl_games | MIT | Native source/wheel notices |
| Spider | CC-by-NC, upstream license | [LICENSE](https://github.com/facebookresearch/spider/blob/71238456bf97a7eeb3d0471aa31974e2d404d4ae/LICENSE) |

These are software notices, not a complete asset-rights inventory. Isaac Gym, Isaac Sim, task datasets, and robot/object assets retain their own terms. The [task-pack inventory](../taskpacks/README.md) and source manifests record the bundled inputs; source-specific attribution for every redistributed asset remains part of release preparation.
