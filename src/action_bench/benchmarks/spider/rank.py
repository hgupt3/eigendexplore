"""Exact retained-rank derivation for SPIDER PCA proposal artifacts."""

from __future__ import annotations

import math

import torch


def retained_rank_from_spectrum(
    explained_variance: torch.Tensor, retained_variance: float
) -> int:
    """Return the smallest prefix reaching the declared retained variance."""

    if not math.isfinite(retained_variance) or not 0.0 < retained_variance <= 1.0:
        raise ValueError("retained variance must be finite and in (0, 1]")
    spectrum = explained_variance.detach().to(dtype=torch.float64, device="cpu")
    if spectrum.ndim != 1 or spectrum.numel() == 0:
        raise ValueError("explained-variance spectrum must be a nonempty vector")
    if not bool(torch.isfinite(spectrum).all()) or bool((spectrum < 0.0).any()):
        raise ValueError("explained-variance spectrum must be finite and nonnegative")
    total = float(spectrum.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("explained-variance spectrum has no positive mass")
    if retained_variance == 1.0:
        return int(spectrum.numel())
    threshold = retained_variance * total
    return int(torch.searchsorted(spectrum.cumsum(0), threshold).item()) + 1
