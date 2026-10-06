"""Exact right-hand Eigen-to-actuator binding for the Spider study."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
import yaml
from torch import nn

from action_bench.artifacts import PCAArtifact
from action_bench.hands import load_hand_layout

# Adapter-owned vendor naming and mimic expansion; the fitted basis stays in
# independent hand coordinates. Every target occurs exactly once.
SPIDER_HOST_JOINT_NAME_MAPS = {
    "allegro": {
        "right_index_base_joint": "joint_0.0",
        "right_index_proximal_joint": "joint_1.0",
        "right_index_medial_joint": "joint_2.0",
        "right_index_distal_joint": "joint_3.0",
        "right_middle_base_joint": "joint_4.0",
        "right_middle_proximal_joint": "joint_5.0",
        "right_middle_medial_joint": "joint_6.0",
        "right_middle_distal_joint": "joint_7.0",
        "right_ring_base_joint": "joint_8.0",
        "right_ring_proximal_joint": "joint_9.0",
        "right_ring_medial_joint": "joint_10.0",
        "right_ring_distal_joint": "joint_11.0",
        "right_thumb_base_joint": "joint_12.0",
        "right_thumb_proximal_joint": "joint_13.0",
        "right_thumb_medial_joint": "joint_14.0",
        "right_thumb_distal_joint": "joint_15.0",
    }
}

SPIDER_HOST_JOINT_EXPANSIONS = {
    "inspire": {
        "right": {
            "R_TH_J1": (("right_thumb_proximal_yaw_joint", 1.0),),
            "R_TH_J2": (
                ("right_thumb_proximal_pitch_joint", 1.0),
                ("right_thumb_intermediate_joint", 1.0),
                ("right_thumb_distal_joint", 1.13),
            ),
            "R_FF_J1": (
                ("right_index_proximal_joint", 1.0),
                ("right_index_intermediate_joint", 1.13),
            ),
            "R_MF_J1": (
                ("right_middle_proximal_joint", 1.0),
                ("right_middle_intermediate_joint", 1.13),
            ),
            "R_RF_J1": (
                ("right_ring_proximal_joint", 1.0),
                ("right_ring_intermediate_joint", 1.08),
            ),
            "R_LF_J1": (
                ("right_pinky_proximal_joint", 1.0),
                ("right_pinky_intermediate_joint", 1.13),
            ),
        }
    },
    "schunk": {
        "right": {
            "right_hand_Thumb_Flexion": (
                ("right_hand_Thumb_Flexion", 1.0),
                ("right_hand_j3", 1.01511),
                ("right_hand_j4", 1.44889),
            ),
            "right_hand_Thumb_Opposition": (
                ("right_hand_Thumb_Opposition", 1.0),
                ("right_hand_block_joint", 1.0),
            ),
            "right_hand_Index_Finger_Distal": (
                ("right_hand_Index_Finger_Distal", 1.0),
                ("right_hand_j14", 1.045),
            ),
            "right_hand_Index_Finger_Proximal": (
                ("right_hand_Index_Finger_Proximal", 1.0),
            ),
            "right_hand_Middle_Finger_Proximal": (
                ("right_hand_Middle_Finger_Proximal", 1.0),
            ),
            "right_hand_Middle_Finger_Distal": (
                ("right_hand_Middle_Finger_Distal", 1.0),
                ("right_hand_j15", 1.0454),
            ),
            "right_hand_Ring_Finger": (
                ("right_hand_Ring_Finger", 1.0),
                ("right_hand_j12", 1.3588),
                ("right_hand_j16", 1.42093),
            ),
            "right_hand_Pinky": (
                ("right_hand_Pinky", 1.0),
                ("right_hand_j13", 1.3588),
                ("right_hand_j17", 1.42307),
            ),
            "right_hand_Finger_Spread": (
                ("right_hand_Finger_Spread", 1.0),
                ("right_hand_index_spread", 0.5),
                ("right_hand_ring_spread", 0.5),
            ),
        }
    },
}


class SpiderEigenProposal(NamedTuple):
    full_delta: torch.Tensor
    hand_delta: torch.Tensor
    coefficients: torch.Tensor


class SpiderEigenProposalAdapter(nn.Module):
    """Decode one canonical-right Eigen head; leave all nonfinger controls zero."""

    @classmethod
    def from_artifact(
        cls, artifact_path, *, control_joint_names, host_name_map, k, device, dtype
    ):
        if not dtype.is_floating_point:
            raise ValueError("Spider Eigen adapter requires a floating dtype")
        manifest = yaml.safe_load((Path(artifact_path) / "manifest.yaml").read_text())
        layout = load_hand_layout(manifest["canonical_hand_layout"])
        if layout.side != "right":
            raise ValueError("Spider requires a canonical-right hand")
        artifact = PCAArtifact.load(artifact_path, hand_layout=layout)
        if host_name_map is not None and host_name_map != layout.family:
            raise ValueError("host mapping differs from artifact family")
        if host_name_map in SPIDER_HOST_JOINT_EXPANSIONS:
            mapping = SPIDER_HOST_JOINT_EXPANSIONS[host_name_map]["right"]
        elif host_name_map in SPIDER_HOST_JOINT_NAME_MAPS:
            names = SPIDER_HOST_JOINT_NAME_MAPS[host_name_map]
            mapping = {source: ((host, 1.0),) for host, source in names.items()}
        elif host_name_map is None:
            mapping = {name: ((name, 1.0),) for name in layout.joint_names}
        else:
            raise ValueError(f"unknown Spider host mapping: {host_name_map}")
        if set(mapping) != set(layout.joint_names):
            raise ValueError("host mapping does not exactly cover artifact joints")
        names = tuple(
            name for source in layout.joint_names for name, _ in mapping[source]
        )
        controls = tuple(control_joint_names)
        if len(set(controls)) != len(controls) or len(set(names)) != len(names):
            raise ValueError("Spider actuator names must be unique")
        if not set(names) <= set(controls):
            raise ValueError("Spider controls are missing mapped finger joints")
        if not 1 <= k <= layout.joint_count:
            raise ValueError("Spider Eigen rank is outside the artifact")
        result = cls()
        result.control_joint_names = controls
        result.hand_layouts = (layout,)
        result.hand_count = 1
        result.control_dim = len(controls)
        result.joint_dim = layout.joint_count
        result.host_joint_dim = len(names)
        result.host_joint_names = (names,)
        result.k = k
        result.artifact_id = artifact.manifest.id
        expansion = torch.zeros(
            1, layout.joint_count, len(names), device=device, dtype=dtype
        )
        for row, source in enumerate(layout.joint_names):
            for target, multiplier in mapping[source]:
                expansion[0, row, names.index(target)] = multiplier
        if not torch.isfinite(expansion).all() or (expansion.abs().sum(1) == 0).any():
            raise ValueError("invalid Spider mimic expansion")
        for name, value in {
            "components": artifact.components[:k].to(device=device, dtype=dtype),
            "component_std": artifact.component_std[:k].to(device=device, dtype=dtype),
            "coordinate_signs": torch.tensor(
                [layout.joint_signs], device=device, dtype=dtype
            ),
            "host_expansion": expansion,
            "finger_control_indices": torch.tensor(
                [controls.index(n) for n in names], device=device
            ),
        }.items():
            result.register_buffer(name, value.detach().clone())
        return result

    def _check(self, value, tail):
        if value.shape[-len(tail) :] != tail:
            raise ValueError(f"Spider tensor must end in {tail}, got {value.shape}")
        if (
            value.device != self.components.device
            or value.dtype != self.components.dtype
        ):
            raise ValueError("Spider tensor device or dtype differs from adapter")

    def forward(self, standard_noise, sample_scale):
        self._check(standard_noise, (1, self.k))
        if standard_noise.ndim != 4 or sample_scale.shape != standard_noise.shape[:2]:
            raise ValueError(
                "Spider noise and sample scales must share particle/knot axes"
            )
        self._check(sample_scale, standard_noise.shape[:2])
        coefficients = (
            standard_noise * sample_scale[..., None, None] * self.component_std
        )
        return self.decode_coefficients(coefficients)

    def expand_canonical_hand_values(self, values):
        self._check(values, (1, self.joint_dim))
        return torch.einsum(
            "...hj,hjm->...hm", values * self.coordinate_signs, self.host_expansion
        )

    def decode_coefficients(self, coefficients):
        self._check(coefficients, (1, self.k))
        canonical = torch.einsum("...hk,kj->...hj", coefficients, self.components)
        hand_delta = self.expand_canonical_hand_values(canonical)
        flat = hand_delta.flatten(start_dim=-2)
        indices = self.finger_control_indices.expand_as(flat)
        full = flat.new_zeros(*flat.shape[:-1], self.control_dim).scatter(
            -1, indices, flat
        )
        return SpiderEigenProposal(full, hand_delta, coefficients)
