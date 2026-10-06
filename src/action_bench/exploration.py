"""Build EigenDExplore settings from packaged bases in explicit action coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from eigendexplore import EigenDExplore

from .catalog import load_basis
from .hands import load_hand_layout

if TYPE_CHECKING:
    from .settings import Study


def load_eigendexplore(
    hand: str,
    *,
    side: str = "right",
    k: int | None = None,
    action_joint_names: Sequence[str] | None = None,
    coordinate_scale: torch.Tensor | Sequence[float] | None = None,
    eigen_rms: float = 0.5**0.5,
    spectrum_power: float = 1.0,
    correlation: float = 0.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EigenDExplore:
    """Load a packaged PCA into any Gaussian policy, without benchmark settings.

    Output columns follow ``action_joint_names`` (default: selected layout).
    Every independent finger joint must occur exactly once; additional actions
    receive no Eigen noise. Names are exact, so translate host names explicitly.
    Signs convert canonical-right components to the selected physical hand.

    Default coordinates are absolute joint targets normalized to [-1, 1].
    For other linear action maps, pass native units per action unit in complete
    action order. RMS is normalized over finger channels, independent of extra
    wrist/arm channels. IID scales remain supplied by the policy; pair the
    defaults with finger IID sigma sqrt(0.5). ``correlation`` > 0 makes the
    Eigen noise persist across steps (see ``EigenDExplore``).
    """
    from .catalog import basis_entry

    entry = basis_entry(hand)
    layout = load_hand_layout(f"{hand}_{side}")
    artifact = load_basis(hand, side=side)
    names = tuple(
        layout.joint_names if action_joint_names is None else action_joint_names
    )
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("action_joint_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("action_joint_names contains duplicates")
    missing = sorted(set(layout.joint_names) - set(names))
    if missing:
        raise ValueError(
            f"action_joint_names is missing independent finger joints: {missing}"
        )
    columns = torch.tensor(
        [names.index(name) for name in layout.joint_names], device=device
    )
    if coordinate_scale is None:
        scale = (
            torch.tensor(layout.joint_upper, device=device, dtype=dtype)
            - torch.tensor(layout.joint_lower, device=device, dtype=dtype)
        ) * 0.5
    else:
        full_scale = torch.as_tensor(coordinate_scale, device=device, dtype=dtype)
        if full_scale.shape != (len(names),) or not bool(
            torch.isfinite(full_scale).all() & (full_scale > 0).all()
        ):
            raise ValueError(
                "coordinate_scale must be positive and finite in complete action order"
            )
        scale = full_scale.index_select(0, columns)
    components = artifact.components.to(device=device, dtype=dtype)
    physical_components = components * torch.tensor(
        layout.joint_signs, device=device, dtype=dtype
    )
    noise = EigenDExplore.from_pca(
        physical_components,
        artifact.component_std.to(device=device, dtype=dtype),
        rank=entry.k if k is None else k,
        eigen_rms=eigen_rms,
        spectrum_power=spectrum_power,
        coordinate_scale=scale,
    )
    basis = noise.basis.new_zeros((noise.basis.shape[0], len(names)))
    basis.index_copy_(1, columns, noise.basis)
    return EigenDExplore(
        basis, noise.log_scale.detach().exp(), correlation=correlation
    )


def hand_noise(
    study: Study, *, side: str = "right"
) -> tuple[torch.Tensor, torch.Tensor]:
    if study.method != "eigendexplore":
        raise ValueError("noise settings require method=eigendexplore")
    layout = load_hand_layout(f"{study.hand}_{side}")
    artifact = load_basis(study.hand, side=side)
    half_range = (
        torch.tensor(layout.joint_upper, dtype=torch.float64)
        - torch.tensor(layout.joint_lower, dtype=torch.float64)
    ) * 0.5
    noise = EigenDExplore.from_pca(
        artifact.components.double(),
        artifact.component_std.double(),
        rank=study.k,
        spectrum_power=study.parameters["spectrum_power"],
        eigen_rms=study.parameters["eigen_rms"],
        coordinate_scale=half_range,
    )
    return noise.basis, noise.log_scale.detach().exp()


def policy_noise(study: Study, *, prefix: int = 0) -> dict:
    """One hand in canonical action order; leading host channels stay at sigma 1."""
    if prefix < 0:
        raise ValueError("prefix must be nonnegative")
    basis, sigma = hand_noise(study)
    full = torch.zeros((basis.shape[0], prefix + basis.shape[1]), dtype=basis.dtype)
    full[:, prefix:] = basis
    iid = torch.ones(full.shape[1], dtype=basis.dtype)
    iid[prefix:] = study.parameters["iid_sigma"]
    return {
        "basis": full.tolist(),
        "eigen_sigma": sigma.tolist(),
        "iid_sigma": iid.tolist(),
    }
