# Fixed benchmark task inputs

These are ordinary Git files included with a clone. They remain outside the Python wheel so the independent EigenDExplore and core packages stay small.

| Archive | Contents | Use |
| --- | --- | --- |
| `spider-oakinkv2-right-10task-v3.tar.gz` | Six hands × ten tasks; scenes, meshes, and reference motions | Extract as shown in [docs/spider.md](../docs/spider.md) |
| `dextreme-allegro-caps-v1.tar.gz` | Five calibrated Allegro URDF/mesh files (56 KB) | Installed by `scripts/setup_benchmark.py dextreme` |

Spider's pack has a `taskpack.json` manifest inside it. The adjacent DeXtreme JSON file inventories its supplement and host revision. These inputs preserve the existing prepared tasks; they are not trained policy checkpoints, raw recordings, or the corpora used to fit Eigen bases. No retargeting step is required to use them.

Task inputs and robot/object assets retain their original source terms. The repository's MIT software license does not relicense them.
