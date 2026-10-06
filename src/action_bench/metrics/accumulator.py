"""Scriptable, GPU-resident action-representation metric accumulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

_NUM_BINS = 64
_FLOAT64_BUFFERS = (
    "joint_range",
    "joint_half_range",
    "posterior_mean_variance",
    "coordinate_span",
    "action_sum",
    "action_square_sum",
    "action_abs_sum",
    "action_sat_sum",
    "action_count",
    "action_outer_sum",
    "action_hist",
    "target_sat_sum",
    "target_smooth_sum",
    "target_track_sum",
    "target_min",
    "target_max",
    "target_count",
    "target_smooth_count",
    "target_track_count",
    "target_hist",
    "base_motion_abs_sum",
    "residual_abs_sum",
    "residual_delta_abs_sum",
    "base_motion_count",
    "residual_count",
    "residual_delta_count",
    "coordinate_sum",
    "coordinate_square_sum",
    "coordinate_sat_sum",
    "coordinate_step_abs_sum",
    "coordinate_step_norm_sum",
    "coordinate_track_abs_sum",
    "coordinate_track_norm_sum",
    "coordinate_min",
    "coordinate_max",
    "coordinate_count",
    "coordinate_step_count",
    "coordinate_track_count",
    "coordinate_hist",
    "onmanifold_square_sum",
    "onmanifold_count",
)


def _as_vector(
    value: Tensor | None,
    *,
    name: str,
    dimension: int | None = None,
) -> Tensor | None:
    if value is None:
        return None
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if dimension is not None and value.shape != (dimension,):
        raise ValueError(f"{name} shape differs from the coordinate dimension")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return value


class HandActionMetrics(nn.Module):
    """Accumulate one hand's action metrics without leaving the rollout device."""

    block_names: List[str]
    block_starts: List[int]
    block_dims: List[int]
    usage_abs_keys: List[str]
    usage_std_keys: List[str]
    usage_sat_keys: List[str]
    effdim_pr_keys: List[str]
    effdim_frac_keys: List[str]
    action_hist_keys: List[str]
    usage_dim_mean_keys: List[str]
    usage_dim_std_keys: List[str]
    usage_dim_abs_keys: List[str]
    usage_dim_sat_keys: List[str]
    usage_dim_block_indices: List[int]
    usage_dim_indices: List[int]
    target_joint_sat_keys: List[str]
    target_joint_smooth_keys: List[str]
    target_joint_coverage_keys: List[str]
    target_joint_track_keys: List[str]
    decomp_joint_keys: List[str]
    latent_dim_sat_keys: List[str]
    latent_dim_step_keys: List[str]
    latent_dim_track_keys: List[str]
    latent_dim_var_keys: List[str]
    latent_dim_var_ratio_keys: List[str]
    pca_dim_coverage_keys: List[str]
    pca_dim_energy_keys: List[str]
    aggregate_keys: List[str]

    __constants__ = [
        "action_dim",
        "joint_dim",
        "num_envs",
        "num_bins",
        "detail_full",
        "has_decomp",
        "has_latent",
        "has_latent_scale",
        "has_pca",
        "has_pca_coverage",
        "has_posterior_variance",
        "coordinate_dim",
        "latent_block_index",
        "key_prefix",
        "block_names",
        "block_starts",
        "block_dims",
        "usage_abs_keys",
        "usage_std_keys",
        "usage_sat_keys",
        "effdim_pr_keys",
        "effdim_frac_keys",
        "action_hist_keys",
        "usage_dim_mean_keys",
        "usage_dim_std_keys",
        "usage_dim_abs_keys",
        "usage_dim_sat_keys",
        "usage_dim_block_indices",
        "usage_dim_indices",
        "target_joint_sat_keys",
        "target_joint_smooth_keys",
        "target_joint_coverage_keys",
        "target_joint_track_keys",
        "decomp_joint_keys",
        "latent_dim_sat_keys",
        "latent_dim_step_keys",
        "latent_dim_track_keys",
        "latent_dim_var_keys",
        "latent_dim_var_ratio_keys",
        "pca_dim_coverage_keys",
        "pca_dim_energy_keys",
        "aggregate_keys",
    ]

    def __init__(
        self,
        action_layout: Sequence[Any],
        joint_names: tuple[str, ...],
        joint_lower: Tensor,
        joint_upper: Tensor,
        num_envs: int,
        *,
        latent_low: Tensor | None = None,
        latent_high: Tensor | None = None,
        latent_scale: Tensor | None = None,
        component_std: Tensor | None = None,
        coefficient_low: Tensor | None = None,
        coefficient_high: Tensor | None = None,
        posterior_mean_variance: Tensor | None = None,
        mean_kl_per_coordinate: Tensor | None = None,
        detail: str = "full",
        key_prefix: str = "action_rep/",
    ) -> None:
        super().__init__()
        if detail not in {"aggregate", "full"}:
            raise ValueError("detail must be 'aggregate' or 'full'")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not key_prefix or not key_prefix.endswith("/"):
            raise ValueError("key_prefix must be nonempty and end in '/'")
        if len(joint_names) == 0 or len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names must be nonempty and unique")
        lower = _as_vector(joint_lower, name="joint_lower")
        upper = _as_vector(
            joint_upper,
            name="joint_upper",
            dimension=len(joint_names),
        )
        assert lower is not None and upper is not None
        if lower.shape != (len(joint_names),):
            raise ValueError("joint_lower shape differs from joint_names")
        if lower.device != upper.device or lower.dtype != upper.dtype:
            raise ValueError("joint limit device and dtype must match")
        if bool(torch.any(upper <= lower)):
            raise ValueError("each joint upper limit must exceed its lower limit")

        block_names: list[str] = []
        block_starts: list[int] = []
        block_dims: list[int] = []
        expected_start = 0
        for raw_block in action_layout:
            if len(raw_block) != 3:
                raise ValueError(
                    "action layout entries must be (name, start, dimension)"
                )
            name = str(raw_block[0])
            start = int(raw_block[1])
            dimension = int(raw_block[2])
            if not name or name in block_names:
                raise ValueError("action block names must be nonempty and unique")
            if start != expected_start or dimension < 1:
                raise ValueError(
                    "action blocks must be positive, ordered, and contiguous"
                )
            block_names.append(name)
            block_starts.append(start)
            block_dims.append(dimension)
            expected_start += dimension
        if not block_names:
            raise ValueError("action_layout must contain at least one block")

        latent_pair = (latent_low is not None, latent_high is not None)
        if latent_pair[0] != latent_pair[1]:
            raise ValueError("latent_low and latent_high must be supplied together")
        coefficient_pair = (
            coefficient_low is not None,
            coefficient_high is not None,
        )
        if coefficient_pair[0] != coefficient_pair[1]:
            raise ValueError(
                "coefficient_low and coefficient_high must be supplied together"
            )
        has_latent = latent_low is not None
        has_pca = component_std is not None or coefficient_low is not None
        if has_latent and has_pca:
            raise ValueError("a metrics core cannot be both latent and PCA")
        if not has_latent and any(
            value is not None
            for value in (
                latent_scale,
                posterior_mean_variance,
                mean_kl_per_coordinate,
            )
        ):
            raise ValueError(
                "latent scale and reference statistics require latent bounds"
            )

        coordinate_dim = 0
        if has_latent:
            assert latent_low is not None and latent_high is not None
            coordinate_dim = latent_low.numel()
        elif component_std is not None:
            coordinate_dim = component_std.numel()
        elif coefficient_low is not None:
            coordinate_dim = coefficient_low.numel()

        latent_low_checked = _as_vector(
            latent_low,
            name="latent_low",
            dimension=coordinate_dim if has_latent else None,
        )
        latent_high_checked = _as_vector(
            latent_high,
            name="latent_high",
            dimension=coordinate_dim if has_latent else None,
        )
        latent_scale_checked = _as_vector(
            latent_scale,
            name="latent_scale",
            dimension=coordinate_dim if has_latent else None,
        )
        component_std_checked = _as_vector(
            component_std,
            name="component_std",
            dimension=coordinate_dim if has_pca else None,
        )
        coefficient_low_checked = _as_vector(
            coefficient_low,
            name="coefficient_low",
            dimension=coordinate_dim if has_pca else None,
        )
        coefficient_high_checked = _as_vector(
            coefficient_high,
            name="coefficient_high",
            dimension=coordinate_dim if has_pca else None,
        )
        posterior_checked = _as_vector(
            posterior_mean_variance,
            name="posterior_mean_variance",
            dimension=coordinate_dim if has_latent else None,
        )
        kl_checked = _as_vector(
            mean_kl_per_coordinate,
            name="mean_kl_per_coordinate",
            dimension=coordinate_dim if has_latent else None,
        )
        coordinate_tensors = (
            latent_low_checked,
            latent_high_checked,
            latent_scale_checked,
            component_std_checked,
            coefficient_low_checked,
            coefficient_high_checked,
            posterior_checked,
            kl_checked,
        )
        for coordinate_tensor in coordinate_tensors:
            if coordinate_tensor is not None and (
                coordinate_tensor.device != lower.device
                or coordinate_tensor.dtype != lower.dtype
            ):
                raise ValueError(
                    "coordinate metadata device and dtype must match joint limits"
                )
        if has_latent:
            assert latent_low_checked is not None
            assert latent_high_checked is not None
            if bool(torch.any(latent_high_checked <= latent_low_checked)):
                raise ValueError("each latent upper bound must exceed its lower bound")
        if has_pca and coefficient_low_checked is not None:
            assert coefficient_high_checked is not None
            if bool(torch.any(coefficient_high_checked <= coefficient_low_checked)):
                raise ValueError(
                    "each coefficient upper bound must exceed its lower bound"
                )
        if component_std_checked is not None and bool(
            torch.any(component_std_checked < 0.0)
        ):
            raise ValueError("component_std must be nonnegative")
        if posterior_checked is not None and bool(torch.any(posterior_checked < 0.0)):
            raise ValueError("posterior_mean_variance must be nonnegative")
        if kl_checked is not None and bool(torch.any(kl_checked < 0.0)):
            raise ValueError("mean_kl_per_coordinate must be nonnegative")

        latent_block_index = -1
        for index, name in enumerate(block_names):
            if name == "latent":
                latent_block_index = index
        if has_latent and (
            latent_block_index < 0 or block_dims[latent_block_index] != coordinate_dim
        ):
            raise ValueError("latent metadata differs from the latent action block")
        if has_pca:
            eigen_dims = [
                block_dims[index]
                for index, name in enumerate(block_names)
                if name in {"eigen", "eigen_fast"}
            ]
            if len(eigen_dims) != 1 or eigen_dims[0] != coordinate_dim:
                raise ValueError("PCA metadata differs from the eigen action block")

        self.action_dim = expected_start
        self.joint_dim = len(joint_names)
        self.num_envs = num_envs
        self.num_bins = _NUM_BINS
        self.detail_full = detail == "full"
        self.has_decomp = "joint_residual" in block_names
        self.has_latent = has_latent
        self.has_latent_scale = latent_scale_checked is not None
        self.has_pca = has_pca
        self.has_pca_coverage = coefficient_low_checked is not None
        self.has_posterior_variance = posterior_checked is not None
        self.coordinate_dim = coordinate_dim
        self.latent_block_index = latent_block_index
        self.key_prefix = key_prefix
        self.block_names = block_names
        self.block_starts = block_starts
        self.block_dims = block_dims

        self.usage_abs_keys = []
        self.usage_std_keys = []
        self.usage_sat_keys = []
        self.effdim_pr_keys = []
        self.effdim_frac_keys = []
        self.action_hist_keys = []
        for name in block_names:
            self.usage_abs_keys.append(key_prefix + "usage/" + name + ".abs_mean")
            self.usage_std_keys.append(key_prefix + "usage/" + name + ".std_mean")
            self.usage_sat_keys.append(key_prefix + "usage/" + name + ".sat")
            self.effdim_pr_keys.append(key_prefix + "effdim/" + name + ".pr")
            self.effdim_frac_keys.append(key_prefix + "effdim/" + name + ".pr_frac")
            self.action_hist_keys.append(key_prefix + "hist/action." + name)

        self.usage_dim_mean_keys = []
        self.usage_dim_std_keys = []
        self.usage_dim_abs_keys = []
        self.usage_dim_sat_keys = []
        self.usage_dim_block_indices = []
        self.usage_dim_indices = []
        if self.detail_full:
            for block_index, name in enumerate(block_names):
                for dimension_index in range(block_dims[block_index]):
                    stem = (
                        key_prefix
                        + "usage_dim/"
                        + name
                        + "."
                        + f"{dimension_index:02d}"
                    )
                    self.usage_dim_mean_keys.append(stem + ".mean")
                    self.usage_dim_std_keys.append(stem + ".std")
                    self.usage_dim_abs_keys.append(stem + ".abs_mean")
                    self.usage_dim_sat_keys.append(stem + ".sat")
                    self.usage_dim_block_indices.append(block_index)
                    self.usage_dim_indices.append(dimension_index)

        sanitized_joint_names = [name.replace("/", ".") for name in joint_names]
        self.target_joint_sat_keys = []
        self.target_joint_smooth_keys = []
        self.target_joint_coverage_keys = []
        self.target_joint_track_keys = []
        self.decomp_joint_keys = []
        if self.detail_full:
            for name in sanitized_joint_names:
                stem = key_prefix + "target_joint/" + name
                self.target_joint_sat_keys.append(stem + ".sat")
                self.target_joint_smooth_keys.append(stem + ".smooth")
                self.target_joint_coverage_keys.append(stem + ".coverage")
                self.target_joint_track_keys.append(stem + ".track_err")
                if self.has_decomp:
                    self.decomp_joint_keys.append(
                        key_prefix + "decomp_joint/" + name + ".residual_abs"
                    )

        self.latent_dim_sat_keys = []
        self.latent_dim_step_keys = []
        self.latent_dim_track_keys = []
        self.latent_dim_var_keys = []
        self.latent_dim_var_ratio_keys = []
        self.pca_dim_coverage_keys = []
        self.pca_dim_energy_keys = []
        if self.detail_full and self.has_latent:
            for index in range(coordinate_dim):
                stem = key_prefix + "latent_dim/z" + str(index)
                self.latent_dim_sat_keys.append(stem + ".sat")
                self.latent_dim_step_keys.append(stem + ".step_rel")
                self.latent_dim_track_keys.append(stem + ".track")
                self.latent_dim_var_keys.append(stem + ".var")
                if self.has_posterior_variance:
                    self.latent_dim_var_ratio_keys.append(stem + ".var_ratio")
        if self.detail_full and self.has_pca:
            for index in range(coordinate_dim):
                stem = key_prefix + "pca_dim/c" + str(index)
                if self.has_pca_coverage:
                    self.pca_dim_coverage_keys.append(stem + ".coverage")
                self.pca_dim_energy_keys.append(stem + ".energy")
        self.aggregate_keys = [
            key_prefix + "target/sat",
            key_prefix + "target/smooth",
            key_prefix + "target/coverage",
            key_prefix + "target/track_err",
            key_prefix + "hist/joint.target",
            key_prefix + "decomp/synergy_share",
            key_prefix + "decomp/base_motion_abs_mean",
            key_prefix + "decomp/residual_abs_mean",
            key_prefix + "latent/sat",
            key_prefix + "latent/z_step",
            key_prefix + "latent/track",
            key_prefix + "latent/track_rel",
            key_prefix + "latent/onmanifold_rmse",
            key_prefix + "latent/coverage",
            key_prefix + "hist/latent.z",
            key_prefix + "_raw/latent_var",
            key_prefix + "pca/coverage",
            key_prefix + "pca/onmanifold_rmse",
            key_prefix + "_raw/pca_var",
        ]

        device = lower.device
        anchor_dtype = lower.dtype
        stats_dtype = torch.float64
        max_block_dim = max(block_dims)
        block_count = len(block_names)

        self.register_buffer(
            "_anchor",
            torch.empty(0, device=device, dtype=anchor_dtype),
            persistent=False,
        )
        self.register_buffer("joint_lower", lower.detach().clone(), persistent=False)
        self.register_buffer("joint_upper", upper.detach().clone(), persistent=False)
        self.register_buffer(
            "joint_range",
            (upper - lower).to(dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "joint_half_range",
            (0.5 * (upper - lower)).to(dtype=stats_dtype),
            persistent=False,
        )

        coordinate_low = torch.zeros(coordinate_dim, device=device, dtype=anchor_dtype)
        coordinate_high = torch.ones(coordinate_dim, device=device, dtype=anchor_dtype)
        if self.has_latent:
            assert latent_low_checked is not None and latent_high_checked is not None
            coordinate_low = latent_low_checked.detach().clone()
            coordinate_high = latent_high_checked.detach().clone()
        elif self.has_pca_coverage:
            assert coefficient_low_checked is not None
            assert coefficient_high_checked is not None
            coordinate_low = coefficient_low_checked.detach().clone()
            coordinate_high = coefficient_high_checked.detach().clone()
        self.register_buffer("coordinate_low", coordinate_low, persistent=False)
        self.register_buffer("coordinate_high", coordinate_high, persistent=False)
        self.register_buffer(
            "coordinate_span",
            (coordinate_high - coordinate_low).to(dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "latent_scale",
            (
                torch.zeros(coordinate_dim, device=device, dtype=anchor_dtype)
                if latent_scale_checked is None
                else latent_scale_checked.detach().clone()
            ),
            persistent=False,
        )
        self.register_buffer(
            "component_std",
            (
                torch.zeros(coordinate_dim, device=device, dtype=anchor_dtype)
                if component_std_checked is None
                else component_std_checked.detach().clone()
            ),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_variance",
            (
                torch.zeros(coordinate_dim, device=device, dtype=stats_dtype)
                if posterior_checked is None
                else posterior_checked.detach().to(dtype=stats_dtype)
            ),
            persistent=False,
        )
        self.register_buffer(
            "mean_kl_per_coordinate",
            (
                torch.zeros(coordinate_dim, device=device, dtype=anchor_dtype)
                if kl_checked is None
                else kl_checked.detach().clone()
            ),
            persistent=False,
        )

        self.register_buffer(
            "action_sum",
            torch.zeros(self.action_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "action_square_sum",
            torch.zeros(self.action_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "action_abs_sum",
            torch.zeros(self.action_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "action_sat_sum",
            torch.zeros(self.action_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "action_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "action_outer_sum",
            torch.zeros(
                (block_count, max_block_dim, max_block_dim),
                device=device,
                dtype=stats_dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_hist",
            torch.zeros((block_count, self.num_bins), device=device, dtype=stats_dtype),
            persistent=False,
        )

        self.register_buffer(
            "target_sat_sum",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_smooth_sum",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_track_sum",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_min",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_max",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_smooth_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_track_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "target_hist",
            torch.zeros(self.num_bins, device=device, dtype=stats_dtype),
            persistent=False,
        )

        self.register_buffer(
            "base_motion_abs_sum",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "residual_abs_sum",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "residual_delta_abs_sum",
            torch.zeros(self.joint_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "base_motion_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "residual_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "residual_delta_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "prev_base",
            torch.zeros((num_envs, self.joint_dim), device=device, dtype=anchor_dtype),
            persistent=False,
        )
        self.register_buffer(
            "prev_residual",
            torch.zeros((num_envs, self.joint_dim), device=device, dtype=anchor_dtype),
            persistent=False,
        )

        self.register_buffer(
            "coordinate_sum",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_square_sum",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_sat_sum",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_step_abs_sum",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_step_norm_sum",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_track_abs_sum",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_track_norm_sum",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_min",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_max",
            torch.zeros(coordinate_dim, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_step_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_track_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "coordinate_hist",
            torch.zeros(self.num_bins, device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "onmanifold_square_sum",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "onmanifold_count",
            torch.zeros((), device=device, dtype=stats_dtype),
            persistent=False,
        )
        self.register_buffer(
            "reset_latch",
            torch.zeros(num_envs, device=device, dtype=torch.bool),
            persistent=False,
        )

    def _apply(self, fn: Any, recurse: bool = True) -> HandActionMetrics:
        super()._apply(fn, recurse=recurse)
        for name in _FLOAT64_BUFFERS:
            value = self._buffers[name]
            assert value is not None
            self._buffers[name] = value.to(dtype=torch.float64)
        return self

    def _check_matrix(self, value: Tensor, dimension: int, name: str) -> None:
        if (
            value.dim() != 2
            or value.size(0) != self.num_envs
            or value.size(1) != dimension
        ):
            raise ValueError(name + " has the wrong shape")
        if value.device != self._anchor.device or value.dtype != self._anchor.dtype:
            raise ValueError(name + " device or dtype differs from the metrics core")

    def _check_mask(self, value: Tensor, name: str) -> None:
        if (
            value.dim() != 1
            or value.size(0) != self.num_envs
            or value.dtype != torch.bool
        ):
            raise ValueError(name + " must be a bool vector with one row per env")
        if value.device != self._anchor.device:
            raise ValueError(name + " device differs from the metrics core")

    def _safe_divide(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        return torch.where(
            denominator > 0.0,
            numerator / torch.clamp_min(denominator, 1.0e-300),
            torch.zeros_like(numerator),
        )

    def _histogram(self, normalized: Tensor) -> Tensor:
        indices = torch.floor(normalized * float(self.num_bins)).to(torch.int64)
        indices = torch.clamp(indices, min=0, max=self.num_bins - 1)
        return torch.bincount(indices.reshape(-1), minlength=self.num_bins).to(
            dtype=torch.float64
        )

    @torch.jit.export
    def notify_reset(self, env_mask: Tensor) -> None:
        self._check_mask(env_mask, "env_mask")
        self.reset_latch.logical_or_(env_mask)
        expanded = env_mask.unsqueeze(1)
        self.prev_base.copy_(
            torch.where(expanded, torch.zeros_like(self.prev_base), self.prev_base)
        )
        self.prev_residual.copy_(
            torch.where(
                expanded, torch.zeros_like(self.prev_residual), self.prev_residual
            )
        )

    @torch.jit.export
    def update_step(
        self,
        action_raw: Tensor,
        action: Tensor,
        joint_target: Tensor,
        prev_joint_target: Tensor,
        base_target: Optional[Tensor],
        residual: Optional[Tensor],
        z_prev: Optional[Tensor],
        z_next: Optional[Tensor],
        invalid_delta_mask: Optional[Tensor],
    ) -> None:
        self._check_matrix(action_raw, self.action_dim, "action_raw")
        self._check_matrix(action, self.action_dim, "action")
        self._check_matrix(joint_target, self.joint_dim, "joint_target")
        self._check_matrix(prev_joint_target, self.joint_dim, "prev_joint_target")
        if (base_target is None) != (residual is None):
            raise ValueError("base_target and residual must be supplied together")
        if base_target is not None:
            if not self.has_decomp:
                raise ValueError(
                    "decomposition tensors require a residual action block"
                )
            self._check_matrix(base_target, self.joint_dim, "base_target")
        if residual is not None:
            self._check_matrix(residual, self.joint_dim, "residual")
        if z_prev is not None:
            self._check_matrix(z_prev, self.coordinate_dim, "z_prev")
        if z_next is not None:
            self._check_matrix(z_next, self.coordinate_dim, "z_next")
        if (z_prev is not None or z_next is not None) and not (
            self.has_latent or self.has_pca
        ):
            raise ValueError("coordinate tensors require latent or PCA metadata")
        if invalid_delta_mask is not None:
            self._check_mask(invalid_delta_mask, "invalid_delta_mask")
            invalid = torch.logical_or(self.reset_latch, invalid_delta_mask)
        else:
            invalid = self.reset_latch.clone()
        valid = torch.logical_not(invalid)
        valid64 = valid.to(dtype=torch.float64).unsqueeze(1)
        valid_count = valid64.sum()
        self.reset_latch.zero_()

        action64 = action.to(dtype=torch.float64)
        raw64 = action_raw.to(dtype=torch.float64)
        self.action_sum.add_(action64.sum(dim=0))
        self.action_square_sum.add_(action64.square().sum(dim=0))
        self.action_abs_sum.add_(action64.abs().sum(dim=0))
        self.action_sat_sum.add_(
            (raw64.abs() >= 1.0).to(dtype=torch.float64).sum(dim=0)
        )
        self.action_count.add_(float(self.num_envs))
        for block_index in range(len(self.block_starts)):
            start = self.block_starts[block_index]
            dimension = self.block_dims[block_index]
            block = action64[:, start : start + dimension]
            self.action_outer_sum[block_index, :dimension, :dimension].add_(
                block.transpose(0, 1) @ block
            )
            normalized_action = 0.5 * (block + 1.0)
            self.action_hist[block_index].add_(self._histogram(normalized_action))

        target64 = joint_target.to(dtype=torch.float64)
        prev_target64 = prev_joint_target.to(dtype=torch.float64)
        epsilon = 1.0e-3 * self.joint_range
        target_sat = torch.logical_or(
            target64 <= self.joint_lower.to(dtype=torch.float64) + epsilon,
            target64 >= self.joint_upper.to(dtype=torch.float64) - epsilon,
        )
        self.target_sat_sum.add_(target_sat.to(dtype=torch.float64).sum(dim=0))
        self.target_smooth_sum.add_(
            (((target64 - prev_target64).abs() / self.joint_range) * valid64).sum(dim=0)
        )
        self.target_smooth_count.add_(valid_count)
        batch_min = torch.amin(target64, dim=0)
        batch_max = torch.amax(target64, dim=0)
        seen = self.target_count > 0.0
        self.target_min.copy_(
            torch.where(seen, torch.minimum(self.target_min, batch_min), batch_min)
        )
        self.target_max.copy_(
            torch.where(seen, torch.maximum(self.target_max, batch_max), batch_max)
        )
        self.target_count.add_(float(self.num_envs))
        normalized_target = (
            target64 - self.joint_lower.to(dtype=torch.float64)
        ) / self.joint_range
        self.target_hist.add_(self._histogram(normalized_target))

        if base_target is not None and residual is not None:
            base_delta64 = (base_target - self.prev_base).abs().to(dtype=torch.float64)
            residual64 = residual.to(dtype=torch.float64)
            residual_delta64 = (
                (residual - self.prev_residual).abs().to(dtype=torch.float64)
            )
            self.base_motion_abs_sum.add_((base_delta64 * valid64).sum(dim=0))
            self.residual_abs_sum.add_(residual64.abs().sum(dim=0))
            self.residual_delta_abs_sum.add_((residual_delta64 * valid64).sum(dim=0))
            self.base_motion_count.add_(valid_count)
            self.residual_count.add_(float(self.num_envs))
            self.residual_delta_count.add_(valid_count)
            self.prev_base.copy_(base_target)
            self.prev_residual.copy_(residual)

        if z_next is not None:
            z_next64 = z_next.to(dtype=torch.float64)
            self.coordinate_sum.add_(z_next64.sum(dim=0))
            self.coordinate_square_sum.add_(z_next64.square().sum(dim=0))
            coord_min = torch.amin(z_next64, dim=0)
            coord_max = torch.amax(z_next64, dim=0)
            coord_seen = self.coordinate_count > 0.0
            self.coordinate_min.copy_(
                torch.where(
                    coord_seen, torch.minimum(self.coordinate_min, coord_min), coord_min
                )
            )
            self.coordinate_max.copy_(
                torch.where(
                    coord_seen, torch.maximum(self.coordinate_max, coord_max), coord_max
                )
            )
            self.coordinate_count.add_(float(self.num_envs))
            if self.has_latent:
                low64 = self.coordinate_low.to(dtype=torch.float64)
                high64 = self.coordinate_high.to(dtype=torch.float64)
                if z_prev is not None and self.has_latent_scale:
                    start = self.block_starts[self.latent_block_index]
                    latent_action = action[:, start : start + self.coordinate_dim]
                    candidate = z_prev + latent_action * self.latent_scale
                    clamp_hit = torch.logical_or(
                        candidate < self.coordinate_low,
                        candidate > self.coordinate_high,
                    )
                else:
                    coordinate_epsilon = 1.0e-3 * self.coordinate_span
                    clamp_hit = torch.logical_or(
                        (z_next64 - low64).abs() <= coordinate_epsilon,
                        (z_next64 - high64).abs() <= coordinate_epsilon,
                    )
                self.coordinate_sat_sum.add_(
                    clamp_hit.to(dtype=torch.float64).sum(dim=0)
                )
                normalized_coordinate = (z_next64 - low64) / self.coordinate_span
                self.coordinate_hist.add_(self._histogram(normalized_coordinate))
            if z_prev is not None:
                dz64 = z_next64 - z_prev.to(dtype=torch.float64)
                self.coordinate_step_abs_sum.add_((dz64.abs() * valid64).sum(dim=0))
                self.coordinate_step_norm_sum.add_(
                    (
                        torch.linalg.vector_norm(dz64, dim=1, ord=2)
                        * valid64.squeeze(1)
                    ).sum()
                )
                self.coordinate_step_count.add_(valid_count)

    @torch.jit.export
    def update_tracking(
        self,
        measured_qpos: Tensor,
        prev_joint_target: Optional[Tensor],
        measured_latent: Optional[Tensor],
        commanded_z: Optional[Tensor],
        reconstruction_error: Optional[Tensor],
    ) -> None:
        self._check_matrix(measured_qpos, self.joint_dim, "measured_qpos")
        if prev_joint_target is not None:
            self._check_matrix(prev_joint_target, self.joint_dim, "prev_joint_target")
            tracking = (measured_qpos - prev_joint_target).abs().to(
                dtype=torch.float64
            ) / self.joint_range
            self.target_track_sum.add_(tracking.sum(dim=0))
            self.target_track_count.add_(float(self.num_envs))
        if (measured_latent is None) != (commanded_z is None):
            raise ValueError(
                "measured_latent and commanded_z must be supplied together"
            )
        if measured_latent is not None and commanded_z is not None:
            if not self.has_latent:
                raise ValueError("latent tracking requires latent metadata")
            self._check_matrix(measured_latent, self.coordinate_dim, "measured_latent")
            self._check_matrix(commanded_z, self.coordinate_dim, "commanded_z")
            error64 = (measured_latent - commanded_z).to(dtype=torch.float64)
            self.coordinate_track_abs_sum.add_(error64.abs().sum(dim=0))
            self.coordinate_track_norm_sum.add_(
                torch.linalg.vector_norm(error64, dim=1, ord=2).sum()
            )
            self.coordinate_track_count.add_(float(self.num_envs))
        if reconstruction_error is not None:
            if not (self.has_latent or self.has_pca):
                raise ValueError("reconstruction_error requires latent or PCA metadata")
            self._check_matrix(
                reconstruction_error, self.joint_dim, "reconstruction_error"
            )
            normalized = (
                reconstruction_error.to(dtype=torch.float64) / self.joint_half_range
            )
            self.onmanifold_square_sum.add_(normalized.square().sum())
            self.onmanifold_count.add_(float(self.num_envs * self.joint_dim))

    @torch.jit.export
    def flush(self) -> Dict[str, Tensor]:
        result = torch.jit.annotate(Dict[str, Tensor], {})
        action_mean = self._safe_divide(self.action_sum, self.action_count)
        action_second = self._safe_divide(self.action_square_sum, self.action_count)
        action_variance = torch.clamp_min(action_second - action_mean.square(), 0.0)
        action_std = torch.sqrt(action_variance)
        action_abs = self._safe_divide(self.action_abs_sum, self.action_count)
        action_sat = self._safe_divide(self.action_sat_sum, self.action_count)
        for block_index in range(len(self.block_starts)):
            start = self.block_starts[block_index]
            dimension = self.block_dims[block_index]
            result[self.usage_abs_keys[block_index]] = action_abs[
                start : start + dimension
            ].mean()
            result[self.usage_std_keys[block_index]] = action_std[
                start : start + dimension
            ].mean()
            result[self.usage_sat_keys[block_index]] = action_sat[
                start : start + dimension
            ].mean()
            outer = self._safe_divide(
                self.action_outer_sum[block_index, :dimension, :dimension],
                self.action_count,
            )
            mean = action_mean[start : start + dimension]
            covariance = outer - mean.unsqueeze(1) * mean.unsqueeze(0)
            trace = torch.trace(covariance)
            denominator = covariance.square().sum()
            pr = self._safe_divide(trace.square(), denominator)
            result[self.effdim_pr_keys[block_index]] = pr
            result[self.effdim_frac_keys[block_index]] = pr / float(dimension)
            result[self.action_hist_keys[block_index]] = self.action_hist[
                block_index
            ].clone()

        target_sat = self._safe_divide(self.target_sat_sum, self.target_count)
        target_smooth = self._safe_divide(
            self.target_smooth_sum, self.target_smooth_count
        )
        target_coverage = self._safe_divide(
            self.target_max - self.target_min, self.joint_range
        )
        target_track = self._safe_divide(self.target_track_sum, self.target_track_count)
        result[self.aggregate_keys[0]] = target_sat.mean()
        result[self.aggregate_keys[1]] = target_smooth.mean()
        result[self.aggregate_keys[2]] = target_coverage.mean()
        result[self.aggregate_keys[3]] = target_track.mean()
        result[self.aggregate_keys[4]] = self.target_hist.clone()

        if self.has_decomp:
            base_raw = self.base_motion_abs_sum.sum()
            residual_delta_raw = self.residual_delta_abs_sum.sum()
            result[self.aggregate_keys[5]] = self._safe_divide(
                base_raw, base_raw + residual_delta_raw
            )
            base_normalized = self._safe_divide(
                self.base_motion_abs_sum / self.joint_range, self.base_motion_count
            )
            residual_normalized = self._safe_divide(
                self.residual_abs_sum / self.joint_range, self.residual_count
            )
            result[self.aggregate_keys[6]] = base_normalized.mean()
            result[self.aggregate_keys[7]] = residual_normalized.mean()

        coordinate_mean = self._safe_divide(self.coordinate_sum, self.coordinate_count)
        coordinate_second = self._safe_divide(
            self.coordinate_square_sum, self.coordinate_count
        )
        coordinate_variance = torch.clamp_min(
            coordinate_second - coordinate_mean.square(), 0.0
        )
        coordinate_coverage = self._safe_divide(
            self.coordinate_max - self.coordinate_min, self.coordinate_span
        )
        onmanifold_rmse = torch.sqrt(
            self._safe_divide(self.onmanifold_square_sum, self.onmanifold_count)
        )
        if self.has_latent:
            coordinate_sat = self._safe_divide(
                self.coordinate_sat_sum, self.coordinate_count
            )
            coordinate_step = self._safe_divide(
                self.coordinate_step_abs_sum, self.coordinate_step_count
            )
            coordinate_track = self._safe_divide(
                self.coordinate_track_abs_sum, self.coordinate_track_count
            )
            result[self.aggregate_keys[8]] = coordinate_sat.mean()
            result[self.aggregate_keys[9]] = self._safe_divide(
                self.coordinate_step_norm_sum, self.coordinate_step_count
            )
            result[self.aggregate_keys[10]] = self._safe_divide(
                self.coordinate_track_norm_sum, self.coordinate_track_count
            )
            result[self.aggregate_keys[11]] = (
                coordinate_track / self.coordinate_span
            ).mean()
            result[self.aggregate_keys[12]] = onmanifold_rmse
            result[self.aggregate_keys[13]] = coordinate_coverage.mean()
            result[self.aggregate_keys[14]] = self.coordinate_hist.clone()
            result[self.aggregate_keys[15]] = coordinate_variance.clone()
        if self.has_pca:
            if self.has_pca_coverage:
                result[self.aggregate_keys[16]] = coordinate_coverage.mean()
            result[self.aggregate_keys[17]] = onmanifold_rmse
            result[self.aggregate_keys[18]] = coordinate_variance.clone()

        if self.detail_full:
            for key_index in range(len(self.usage_dim_mean_keys)):
                block_index = self.usage_dim_block_indices[key_index]
                action_index = (
                    self.block_starts[block_index] + self.usage_dim_indices[key_index]
                )
                result[self.usage_dim_mean_keys[key_index]] = action_mean[action_index]
                result[self.usage_dim_std_keys[key_index]] = action_std[action_index]
                result[self.usage_dim_abs_keys[key_index]] = action_abs[action_index]
                result[self.usage_dim_sat_keys[key_index]] = action_sat[action_index]
            for joint_index in range(self.joint_dim):
                result[self.target_joint_sat_keys[joint_index]] = target_sat[
                    joint_index
                ]
                result[self.target_joint_smooth_keys[joint_index]] = target_smooth[
                    joint_index
                ]
                result[self.target_joint_coverage_keys[joint_index]] = target_coverage[
                    joint_index
                ]
                result[self.target_joint_track_keys[joint_index]] = target_track[
                    joint_index
                ]
                if self.has_decomp:
                    residual_joint = self._safe_divide(
                        self.residual_abs_sum[joint_index]
                        / self.joint_range[joint_index],
                        self.residual_count,
                    )
                    result[self.decomp_joint_keys[joint_index]] = residual_joint
            if self.has_latent:
                coordinate_sat = self._safe_divide(
                    self.coordinate_sat_sum, self.coordinate_count
                )
                coordinate_step = self._safe_divide(
                    self.coordinate_step_abs_sum, self.coordinate_step_count
                )
                coordinate_track = self._safe_divide(
                    self.coordinate_track_abs_sum, self.coordinate_track_count
                )
                for coordinate_index in range(self.coordinate_dim):
                    result[self.latent_dim_sat_keys[coordinate_index]] = coordinate_sat[
                        coordinate_index
                    ]
                    result[self.latent_dim_step_keys[coordinate_index]] = (
                        coordinate_step[coordinate_index]
                        / self.coordinate_span[coordinate_index]
                    )
                    result[self.latent_dim_track_keys[coordinate_index]] = (
                        coordinate_track[coordinate_index]
                    )
                    result[self.latent_dim_var_keys[coordinate_index]] = (
                        coordinate_variance[coordinate_index]
                    )
                    if self.has_posterior_variance:
                        result[self.latent_dim_var_ratio_keys[coordinate_index]] = (
                            self._safe_divide(
                                coordinate_variance[coordinate_index],
                                self.posterior_mean_variance[coordinate_index],
                            )
                        )
            if self.has_pca:
                for coordinate_index in range(self.coordinate_dim):
                    if self.has_pca_coverage:
                        result[self.pca_dim_coverage_keys[coordinate_index]] = (
                            coordinate_coverage[coordinate_index]
                        )
                    result[self.pca_dim_energy_keys[coordinate_index]] = (
                        coordinate_variance[coordinate_index]
                    )

        self.action_sum.zero_()
        self.action_square_sum.zero_()
        self.action_abs_sum.zero_()
        self.action_sat_sum.zero_()
        self.action_count.zero_()
        self.action_outer_sum.zero_()
        self.action_hist.zero_()
        self.target_sat_sum.zero_()
        self.target_smooth_sum.zero_()
        self.target_track_sum.zero_()
        self.target_min.zero_()
        self.target_max.zero_()
        self.target_count.zero_()
        self.target_smooth_count.zero_()
        self.target_track_count.zero_()
        self.target_hist.zero_()
        self.base_motion_abs_sum.zero_()
        self.residual_abs_sum.zero_()
        self.residual_delta_abs_sum.zero_()
        self.base_motion_count.zero_()
        self.residual_count.zero_()
        self.residual_delta_count.zero_()
        self.coordinate_sum.zero_()
        self.coordinate_square_sum.zero_()
        self.coordinate_sat_sum.zero_()
        self.coordinate_step_abs_sum.zero_()
        self.coordinate_step_norm_sum.zero_()
        self.coordinate_track_abs_sum.zero_()
        self.coordinate_track_norm_sum.zero_()
        self.coordinate_min.zero_()
        self.coordinate_max.zero_()
        self.coordinate_count.zero_()
        self.coordinate_step_count.zero_()
        self.coordinate_track_count.zero_()
        self.coordinate_hist.zero_()
        self.onmanifold_square_sum.zero_()
        self.onmanifold_count.zero_()
        return result


def _spearman(first: Tensor, second: Tensor) -> float:
    if first.numel() != second.numel():
        raise ValueError("Spearman inputs must have identical lengths")
    if first.numel() < 2:
        return 0.0
    if bool(torch.all(first == first[0])) or bool(torch.all(second == second[0])):
        return 0.0
    first_rank = torch.argsort(torch.argsort(first)).to(dtype=torch.float64)
    second_rank = torch.argsort(torch.argsort(second)).to(dtype=torch.float64)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = torch.sqrt(
        first_centered.square().sum() * second_centered.square().sum()
    )
    if float(denominator) == 0.0:
        return 0.0
    return float((first_centered * second_centered).sum() / denominator)


def finalize_flush(
    flushed: Dict[str, Tensor],
    *,
    mean_kl_per_coordinate: Tensor | Sequence[float] | None = None,
    component_std: Tensor | Sequence[float] | None = None,
) -> dict[str, float | tuple[list[int], list[float]]]:
    """Transfer one flushed window and convert it to logger-ready Python data."""

    entries = list(flushed.items())
    tensors = [value.reshape(-1) for _, value in entries]
    reference_names: list[str] = []
    reference_tensors: list[Tensor] = []
    if entries:
        device = entries[0][1].device
        if mean_kl_per_coordinate is not None:
            reference_names.append("mean_kl_per_coordinate")
            reference_tensors.append(
                torch.as_tensor(mean_kl_per_coordinate, device=device).reshape(-1)
            )
        if component_std is not None:
            reference_names.append("component_std")
            reference_tensors.append(
                torch.as_tensor(component_std, device=device).reshape(-1)
            )
    lengths = [tensor.numel() for tensor in tensors]
    reference_lengths = [tensor.numel() for tensor in reference_tensors]
    if tensors or reference_tensors:
        packed = torch.cat(tensors + reference_tensors).detach().cpu()
    else:
        packed = torch.empty(0, dtype=torch.float64)

    unpacked: dict[str, Tensor] = {}
    offset = 0
    for (key, _), length in zip(entries, lengths, strict=True):
        unpacked[key] = packed[offset : offset + length]
        offset += length
    references: dict[str, Tensor] = {}
    for name, length in zip(reference_names, reference_lengths, strict=True):
        references[name] = packed[offset : offset + length]
        offset += length

    finalized: dict[str, float | tuple[list[int], list[float]]] = {}
    for key, value in unpacked.items():
        if "/_raw/" in key:
            continue
        if "/hist/" in key:
            counts = [int(round(item)) for item in value.tolist()]
            if key.endswith("hist/latent.z") or key.endswith("hist/joint.target"):
                edges = torch.linspace(0.0, 1.0, _NUM_BINS + 1).tolist()
            else:
                edges = torch.linspace(-1.0, 1.0, _NUM_BINS + 1).tolist()
            finalized[key] = (counts, edges)
        else:
            finalized[key] = float(value[0])

    for key, variance in unpacked.items():
        if key.endswith("_raw/latent_var") and "mean_kl_per_coordinate" in references:
            prefix = key[: -len("_raw/latent_var")]
            finalized[prefix + "latent/prior_corr"] = _spearman(
                variance,
                references["mean_kl_per_coordinate"],
            )
        if key.endswith("_raw/pca_var") and "component_std" in references:
            prefix = key[: -len("_raw/pca_var")]
            finalized[prefix + "pca/energy_corr"] = _spearman(
                variance,
                references["component_std"].square(),
            )
    return finalized
