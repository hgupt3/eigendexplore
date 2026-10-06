"""DeXtreme Allegro host names mapped to canonical action-bench names."""

from __future__ import annotations

from collections.abc import Sequence

from action_bench import HandBinding, HandLayout

DEXTREME_ALLEGRO_NAME_MAP: tuple[tuple[str, str], ...] = tuple(
    (
        f"{finger}_joint_{finger_joint}",
        f"joint_{canonical_offset + finger_joint}.0",
    )
    for finger, canonical_offset in (
        ("index", 0),
        ("middle", 4),
        ("ring", 8),
        ("thumb", 12),
    )
    for finger_joint in range(4)
)

DEXTREME_ALLEGRO_HOST_JOINT_NAMES: tuple[str, ...] = tuple(
    host_name for host_name, _ in DEXTREME_ALLEGRO_NAME_MAP
)


def bind_dextreme_allegro_layout(
    hand_layout: HandLayout,
    host_joint_names: Sequence[str] = DEXTREME_ALLEGRO_HOST_JOINT_NAMES,
) -> HandBinding:
    """Bind exact DeXtreme names/order after applying the explicit name map."""

    if hand_layout.id != "allegro_right":
        raise ValueError("DeXtreme requires the canonical 'allegro_right' hand layout")
    if hand_layout.family != "allegro" or hand_layout.side != "right":
        raise ValueError("DeXtreme requires the canonical right Allegro hand layout")

    names = tuple(host_joint_names)
    if len(set(names)) != len(names):
        raise ValueError("DeXtreme host joint names contain duplicates")
    host_to_canonical = dict(DEXTREME_ALLEGRO_NAME_MAP)
    missing = sorted(set(host_to_canonical) - set(names))
    unexpected = sorted(set(names) - set(host_to_canonical))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            "DeXtreme Allegro host joint names are not an exact match: "
            + ", ".join(details)
        )

    mapped_names = tuple(host_to_canonical[name] for name in names)
    if set(host_to_canonical.values()) != set(hand_layout.joint_names):
        raise ValueError(
            "DeXtreme Allegro name-map targets differ from the canonical layout"
        )
    return hand_layout.bind(mapped_names)
