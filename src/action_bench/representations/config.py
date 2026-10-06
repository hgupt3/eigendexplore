"""Reusable typed representation configuration fragments."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class JointRepresentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["joint"]


class PCARepresentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["pca"]
    k: int = Field(gt=0)


RepresentationConfig = Annotated[
    JointRepresentationConfig | PCARepresentationConfig,
    Field(discriminator="type"),
]
