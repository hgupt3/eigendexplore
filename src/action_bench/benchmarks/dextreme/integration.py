"""Export validated target transforms for DeXtreme's separate Python runtime."""

import json
from pathlib import Path

import torch

from action_bench.catalog import basis_entry
from action_bench.settings import Study, build_study_hand

from .bundle import DextremeTorchScriptBundle
from .layout import DEXTREME_ALLEGRO_HOST_JOINT_NAMES, bind_dextreme_allegro_layout


def export_dextreme_bundle(study: Study, output: str | Path, *, num_envs: int):
    if study.benchmark != "dextreme" or num_envs < 1:
        raise ValueError("export requires a DeXtreme study and positive num_envs")
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    runtime = build_study_hand(study, side="right")
    binding = bind_dextreme_allegro_layout(runtime.hand_layout)
    action_type = study.action_config().type
    bundle = DextremeTorchScriptBundle(runtime, binding, action_type, num_envs).eval()
    compiled = torch.jit.script(bundle)
    entry = basis_entry(study.hand) if study.k is not None else None
    manifest = {
        "schema_version": 5,
        "format": "action_bench.dextreme.torchscript.v5",
        "study": study.model_dump(mode="json"),
        "action_type": action_type,
        "action_dim": bundle.action_dim,
        "measured_latent_dim": bundle.measured_latent_dim,
        "commanded_target_obs_dim": bundle.commanded_obs_dim,
        "latent_obs_dim": bundle.measured_latent_dim + bundle.commanded_obs_dim,
        "semantics": "delta" if bundle.commanded_obs_dim else "absolute",
        "host_joint_names": list(DEXTREME_ALLEGRO_HOST_JOINT_NAMES),
        "artifact_id": entry.artifact_id if entry else None,
        "num_envs": num_envs,
    }
    # Compare the exported map to the eager core before publishing.
    generator = torch.Generator().manual_seed(7)
    actions = torch.rand((8, bundle.action_dim), generator=generator) * 2 - 1
    previous = torch.rand((8, bundle.joint_dim), generator=generator)
    previous = runtime.action_space.joint_lower + previous * (
        runtime.action_space.joint_upper - runtime.action_space.joint_lower
    )
    expected = (
        runtime.action_space.decode(actions, previous)
        if bundle.commanded_obs_dim
        else runtime.action_space.decode(actions)
    )
    torch.testing.assert_close(
        compiled.decode(actions, binding.unpack(previous)), binding.unpack(expected)
    )
    root.mkdir(parents=True)
    compiled.save(str(root / "bundle.pt"))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
