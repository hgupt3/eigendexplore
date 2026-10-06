"""Strict schema and loading for local PCA artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors.torch import load_file

from action_bench._io import read_manifest
from action_bench.hands import HandLayout


class PCAFitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: str = Field(min_length=1)
    split_protocol: str = Field(default="custom_explicit", min_length=1)
    objective: Literal[
        "posture_covariance",
        "motion_tangent_covariance",
    ] = "posture_covariance"
    measure: Literal[
        "frame",
        "normalized_joint_arc_length",
    ] = "frame"
    weighting: Literal[
        "equal_trajectory",
        "equal_frame",
        "explicit_dataset_then_trajectory",
    ]
    motion_horizon_ms: int = Field(default=100, gt=0)
    dataset_weights: tuple[float, ...] | None = None
    lower_quantile: float = Field(ge=0.0, lt=0.5)
    upper_quantile: float = Field(gt=0.5, le=1.0)

    @model_validator(mode="after")
    def validate_protocol(self) -> PCAFitConfig:
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be less than upper_quantile")
        explicit_dataset_weighting = (
            self.weighting == "explicit_dataset_then_trajectory"
        )
        if explicit_dataset_weighting != (self.dataset_weights is not None):
            raise ValueError(
                "dataset_weights is required only for explicit_dataset_then_trajectory"
            )
        if self.dataset_weights is not None:
            if not self.dataset_weights or any(
                not math.isfinite(weight) or weight <= 0.0
                for weight in self.dataset_weights
            ):
                raise ValueError("dataset_weights must contain only positive values")
            if not math.isclose(
                sum(self.dataset_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("dataset_weights must sum to one")
        if self.objective == "motion_tangent_covariance" and self.measure != (
            "normalized_joint_arc_length"
        ):
            raise ValueError(
                "motion_tangent_covariance requires normalized_joint_arc_length"
            )
        if self.weighting == "equal_frame" and (
            self.measure != "frame" or self.objective != "posture_covariance"
        ):
            raise ValueError(
                "equal_frame weighting requires frame measure and "
                "posture_covariance objective"
            )
        return self


class PCAFitSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_trajectory_count: int = Field(gt=0)
    source_frame_count: int = Field(gt=0)
    source_sequence_count: int = Field(gt=0)
    fit_trajectory_count: int = Field(gt=0)
    fit_frame_count: int = Field(gt=0)
    fit_source_sequence_count: int = Field(gt=0)


class PCADeploymentLayout(BaseModel):
    """Exact host-name binding for one semantic PCA coordinate layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hand_layout: str = Field(min_length=1)
    joint_names: tuple[str, ...] = Field(min_length=1)
    joint_signs: tuple[Literal[-1, 1], ...]

    @model_validator(mode="after")
    def validate_joint_names(self) -> PCADeploymentLayout:
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("deployment joint_names contains duplicates")
        if len(self.joint_signs) != len(self.joint_names):
            raise ValueError(
                "deployment joint_names and joint_signs must have identical lengths"
            )
        return self


class PCAArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[4]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    kind: Literal["pca"]
    producer_revision: str = Field(min_length=1)
    construction: Literal["single_layout", "mirrored_pair"]
    canonical_hand_layout: str = Field(min_length=1)
    deployment_layouts: tuple[PCADeploymentLayout, ...] = Field(min_length=1)
    dataset: str = Field(min_length=1)
    source_corpora: tuple[str, ...] = Field(min_length=1)
    source_validation_protocols: tuple[str, ...] = Field(min_length=1)
    joint_names: tuple[str, ...] = Field(min_length=1)
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    component_count: int = Field(gt=0)
    fit: PCAFitConfig
    fit_summary: PCAFitSummary
    payload: Literal["pca.safetensors"]

    @model_validator(mode="after")
    def validate_domain(self) -> PCAArtifactManifest:
        deployment_ids = tuple(
            deployment.hand_layout for deployment in self.deployment_layouts
        )
        if len(set(deployment_ids)) != len(deployment_ids):
            raise ValueError("deployment_layouts contains duplicate hand layouts")
        if self.canonical_hand_layout not in deployment_ids:
            raise ValueError("canonical_hand_layout must be one of deployment_layouts")
        canonical = next(
            item
            for item in self.deployment_layouts
            if item.hand_layout == self.canonical_hand_layout
        )
        if canonical.joint_names != self.joint_names:
            raise ValueError(
                "canonical deployment joint_names differ from artifact joint_names"
            )
        expected_layout_count = 1 if self.construction == "single_layout" else 2
        if len(self.deployment_layouts) != expected_layout_count:
            raise ValueError(
                f"{self.construction} PCA requires {expected_layout_count} "
                "deployment hand layout(s)"
            )
        if len(set(self.source_corpora)) != len(self.source_corpora):
            raise ValueError("source_corpora contains duplicates")
        if len(self.source_validation_protocols) != len(self.source_corpora):
            raise ValueError(
                "source_validation_protocols must align with source_corpora"
            )
        joint_count = len(self.joint_names)
        if len(self.joint_lower) != joint_count or len(self.joint_upper) != joint_count:
            raise ValueError(
                "joint_names, joint_lower, and joint_upper must have identical lengths"
            )
        if self.component_count != joint_count:
            raise ValueError("full-rank PCA component_count must equal joint count")
        for deployment in self.deployment_layouts:
            if len(deployment.joint_names) != joint_count:
                raise ValueError(
                    f"deployment layout {deployment.hand_layout!r} has "
                    f"{len(deployment.joint_names)} joints; expected {joint_count}"
                )
        for lower, upper in zip(self.joint_lower, self.joint_upper, strict=True):
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("PCA manifest contains invalid joint limits")
        return self


@dataclass(frozen=True, slots=True)
class PCAArtifact:
    manifest: PCAArtifactManifest
    mean: torch.Tensor
    components: torch.Tensor
    coefficient_low: torch.Tensor
    coefficient_high: torch.Tensor
    component_std: torch.Tensor
    explained_variance: torch.Tensor

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        hand_layout: HandLayout,
    ) -> PCAArtifact:
        root = Path(directory).expanduser().resolve()
        manifest = read_manifest(root / "manifest.yaml", PCAArtifactManifest)
        assert isinstance(manifest, PCAArtifactManifest)
        tensors = load_file(root / manifest.payload, device="cpu")
        required = {
            "mean",
            "components",
            "coefficient_low",
            "coefficient_high",
            "component_std",
            "explained_variance",
        }
        if set(tensors) != required:
            raise ValueError(
                f"PCA payload fields must be {sorted(required)}, got {sorted(tensors)}"
            )
        artifact = cls(manifest=manifest, **tensors)
        artifact.validate(hand_layout=hand_layout)
        return artifact

    def validate(self, *, hand_layout: HandLayout) -> None:
        matches = tuple(
            deployment
            for deployment in self.manifest.deployment_layouts
            if deployment.hand_layout == hand_layout.id
        )
        if not matches:
            raise ValueError(
                f"PCA artifact does not deploy to {hand_layout.id!r}; "
                "expected one of "
                f"{tuple(item.hand_layout for item in self.manifest.deployment_layouts)}"
            )
        deployment = matches[0]
        if deployment.joint_names != hand_layout.joint_names:
            raise ValueError("PCA artifact joint order differs from HandLayout order")
        if deployment.joint_signs != hand_layout.joint_signs:
            raise ValueError(
                "PCA artifact coordinate signs differ from HandLayout signs"
            )
        component_count = self.manifest.component_count
        joint_count = hand_layout.joint_count
        expected = {
            "mean": (joint_count,),
            "components": (component_count, joint_count),
            "coefficient_low": (component_count,),
            "coefficient_high": (component_count,),
            "component_std": (component_count,),
            "explained_variance": (component_count,),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tensor.shape != shape:
                raise ValueError(
                    f"PCA {name} must have shape {shape}, got {tuple(tensor.shape)}"
                )
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"PCA {name} must be finite floating point")
        if bool(torch.any(self.coefficient_low > 0.0)) or bool(
            torch.any(self.coefficient_high < 0.0)
        ):
            raise ValueError("PCA coefficient bounds must contain zero")
        if bool(torch.any(self.coefficient_high < self.coefficient_low)):
            raise ValueError("PCA coefficient bounds are reversed")
        if bool(torch.any(self.component_std < 0.0)):
            raise ValueError("PCA component_std must be nonnegative")
        if bool(torch.any(self.explained_variance < 0.0)):
            raise ValueError("PCA explained_variance must be nonnegative")
        gram = self.components.to(torch.float64) @ self.components.to(torch.float64).T
        identity = torch.eye(component_count, dtype=torch.float64)
        if not torch.allclose(gram, identity, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError("PCA components are not orthonormal")
