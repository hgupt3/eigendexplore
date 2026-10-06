"""SimToolReal Sharpa host names mapped to canonical action-bench names."""

from __future__ import annotations

from collections.abc import Sequence

from action_bench import HandBinding, HandLayout

# The host order is the real Isaac Lab articulation order: breadth-first across
# the five finger chains. The numbered prefixes are present only on the thumb
# CMC flexion joint, the three finger MCP flexion joints, and the pinky CMC
# joint.
SIMTOOLREAL_SHARPA_NAME_MAP: tuple[tuple[str, str], ...] = (
    ("left_1_thumb_CMC_FE", "left_thumb_CMC_FE"),
    ("left_2_index_MCP_FE", "left_index_MCP_FE"),
    ("left_3_middle_MCP_FE", "left_middle_MCP_FE"),
    ("left_4_ring_MCP_FE", "left_ring_MCP_FE"),
    ("left_5_pinky_CMC", "left_pinky_CMC"),
    ("left_thumb_CMC_AA", "left_thumb_CMC_AA"),
    ("left_index_MCP_AA", "left_index_MCP_AA"),
    ("left_middle_MCP_AA", "left_middle_MCP_AA"),
    ("left_ring_MCP_AA", "left_ring_MCP_AA"),
    ("left_pinky_MCP_FE", "left_pinky_MCP_FE"),
    ("left_thumb_MCP_FE", "left_thumb_MCP_FE"),
    ("left_index_PIP", "left_index_PIP"),
    ("left_middle_PIP", "left_middle_PIP"),
    ("left_ring_PIP", "left_ring_PIP"),
    ("left_pinky_MCP_AA", "left_pinky_MCP_AA"),
    ("left_thumb_MCP_AA", "left_thumb_MCP_AA"),
    ("left_index_DIP", "left_index_DIP"),
    ("left_middle_DIP", "left_middle_DIP"),
    ("left_ring_DIP", "left_ring_DIP"),
    ("left_pinky_PIP", "left_pinky_PIP"),
    ("left_thumb_IP", "left_thumb_IP"),
    ("left_pinky_DIP", "left_pinky_DIP"),
)

SIMTOOLREAL_SHARPA_HOST_JOINT_NAMES: tuple[str, ...] = tuple(
    host_name for host_name, _ in SIMTOOLREAL_SHARPA_NAME_MAP
)


def bind_simtoolreal_sharpa_layout(
    hand_layout: HandLayout,
    host_joint_names: Sequence[str] = SIMTOOLREAL_SHARPA_HOST_JOINT_NAMES,
) -> HandBinding:
    """Bind exact SimToolReal names/order through the explicit name map."""

    if hand_layout.id != "sharpa_left":
        raise ValueError("SimToolReal requires the canonical 'sharpa_left' hand layout")
    if hand_layout.family != "sharpa" or hand_layout.side != "left":
        raise ValueError("SimToolReal requires the canonical left Sharpa hand layout")

    names = tuple(host_joint_names)
    if len(set(names)) != len(names):
        raise ValueError("SimToolReal host joint names contain duplicates")
    host_to_canonical = dict(SIMTOOLREAL_SHARPA_NAME_MAP)
    missing = sorted(set(host_to_canonical) - set(names))
    unexpected = sorted(set(names) - set(host_to_canonical))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            "SimToolReal Sharpa host joint names are not an exact match: "
            + ", ".join(details)
        )

    if set(host_to_canonical.values()) != set(hand_layout.joint_names):
        raise ValueError(
            "SimToolReal Sharpa name-map targets differ from the canonical layout"
        )
    mapped_names = tuple(host_to_canonical[name] for name in names)
    return hand_layout.bind(mapped_names)
