"""Six methods with explicit benchmark defaults and delivered-rate units."""

import math
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .action_spaces import (
    EigenAbsoluteConfig,
    EigenDeltaJointDeltaConfig,
    JointAbsoluteConfig,
    JointAbsoluteEigenResidualConfig,
    JointDeltaConfig,
)
from .catalog import basis_entry

Method = Literal[
    "joint_absolute",
    "joint_delta",
    "eigen_absolute",
    "eigen_delta_joint_delta",
    "joint_absolute_eigen_residual",
    "eigendexplore",
]


def benchmark_defaults(benchmark: str) -> dict:
    if benchmark not in ("dextreme", "simtoolreal"):
        raise ValueError(f"unknown benchmark {benchmark!r}")
    return yaml.safe_load(
        files("action_bench").joinpath(f"presets/{benchmark}.yaml").read_text()
    )


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark: Literal["dextreme", "simtoolreal"]
    hand: str
    method: Method
    k: int | None = Field(default=None, gt=0)
    parameters: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve(self):
        defaults = benchmark_defaults(self.benchmark)
        if self.hand not in defaults["hands"]:
            raise ValueError(f"{self.hand} is not supported by {self.benchmark}")
        if self.method not in defaults["methods"]:
            raise ValueError(f"{self.method} is not supported by {self.benchmark}")
        expected = defaults["methods"][self.method]
        if set(self.parameters) - set(expected):
            raise ValueError(
                f"unexpected parameters for {self.method}: {set(self.parameters) - set(expected)}"
            )
        params = {**expected, **self.parameters}
        for name, value in params.items():
            if (
                not math.isfinite(value)
                or value < 0
                or (
                    value == 0
                    and name not in {"spectrum_power", "eigen_residual_fraction"}
                )
            ):
                raise ValueError(f"invalid parameter {name}")
        object.__setattr__(self, "parameters", params)
        if self.method in {"joint_absolute", "joint_delta"}:
            if self.k is not None:
                raise ValueError("joint methods do not use a rank")
        else:
            entry = basis_entry(self.hand)
            from .hands import load_hand_layout

            k = self.k if self.k is not None else entry.k
            if k > load_hand_layout(self.hand + "_right").joint_count:
                raise ValueError("rank exceeds independent joint count")
            object.__setattr__(self, "k", k)
        return self

    def action_config(self):
        """Compile delivered rates into host per-step action scales."""
        defaults = benchmark_defaults(self.benchmark)
        denominator = defaults["control_hz"] * defaults["target_ema"]
        p = self.parameters
        if self.method in ("joint_absolute", "eigendexplore"):
            return JointAbsoluteConfig(type="joint_absolute")
        if self.method == "joint_delta":
            return JointDeltaConfig(
                type=self.method,
                joint_scale={
                    "unit": "joint_range_fraction",
                    "value": p["joint_rate"] / denominator,
                },
            )
        if self.method == "eigen_absolute":
            return EigenAbsoluteConfig(type=self.method)
        if self.method == "eigen_delta_joint_delta":
            return EigenDeltaJointDeltaConfig(
                type=self.method,
                eigen_scale={
                    "unit": "coefficient_span_fraction",
                    "value": p["eigen_rate"] / denominator,
                },
                joint_scale={
                    "unit": "joint_range_fraction",
                    "value": p["joint_rate"] / denominator,
                },
            )
        return JointAbsoluteEigenResidualConfig(
            type=self.method,
            residual_scale={
                "unit": "decoded_offset_fraction",
                "value": p["eigen_residual_fraction"],
            },
        )


def load_study(path: str | Path) -> Study:
    return Study.model_validate(yaml.safe_load(Path(path).read_text()))


def build_study_hand(study: Study, *, side: str, device="cpu", dtype=None):
    import torch

    from .hands import load_hand_layout
    from .representations.config import (
        JointRepresentationConfig,
        PCARepresentationConfig,
    )
    from .runtime import build_runtime_hand

    layout = load_hand_layout(f"{study.hand}_{side}")
    eigen_action = study.method not in {
        "joint_absolute",
        "joint_delta",
        "eigendexplore",
    }
    entry = basis_entry(study.hand) if eigen_action else None
    representation = (
        PCARepresentationConfig(type="pca", k=study.k)
        if eigen_action
        else JointRepresentationConfig(type="joint")
    )
    return build_runtime_hand(
        hand_layout=layout,
        representation=representation,
        action_space=study.action_config(),
        artifact_path=entry.path if entry else None,
        expected_artifact_id=entry.artifact_id if entry else None,
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )
