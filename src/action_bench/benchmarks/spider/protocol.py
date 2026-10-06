"""Paired IID and EigenDExplore cold-start jobs for one hand."""

import itertools
import json
from functools import lru_cache
from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from action_bench.artifacts import PCAArtifact
from action_bench.catalog import basis_entry
from action_bench.hands import load_hand_layout

from .rank import retained_rank_from_spectrum

PACK_ID = "spider-oakinkv2-right-10task-v3"
SETTING_NAMES = ("basis", "retained_variance", "iid_scale", "eigen_scale")


@lru_cache(maxsize=None)
def defaults():
    return yaml.safe_load(
        files("action_bench").joinpath("presets/spider.yaml").read_text()
    )


def hand_binding(hand):
    values = yaml.safe_load(
        files(__package__).joinpath("configs/hands.yaml").read_text()
    )["hands"]
    if hand not in values:
        raise ValueError(f"unsupported Spider hand {hand!r}")
    result = dict(values[hand])
    result["layout_object"] = load_hand_layout(result["layout"])
    return result


@lru_cache(maxsize=None)
def spider_basis(hand):
    if hand not in defaults()["hands"]:
        raise ValueError(f"unsupported Spider hand {hand!r}")
    return basis_entry(hand)


@lru_cache(maxsize=None)
def rank(hand, retained):
    entry = spider_basis(hand)
    artifact = PCAArtifact.load(
        entry.path, hand_layout=load_hand_layout(hand + "_right")
    )
    if artifact.manifest.id != entry.artifact_id:
        raise ValueError("Spider basis differs from the packaged catalog identity")
    return retained_rank_from_spectrum(artifact.explained_variance, retained)


@lru_cache(maxsize=None)
def task_names():
    return tuple(
        json.loads(files(__package__).joinpath("configs/tasks.json").read_text())[
            "tasks"
        ]
    )


class Setting(BaseModel):
    """One EigenDExplore sampling setting; IID scale 0 is Eigen noise alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    basis: Literal["human", "random"]
    retained_variance: float = Field(gt=0, le=1, allow_inf_nan=False)
    iid_scale: float = Field(ge=0, allow_inf_nan=False)
    eigen_scale: float = Field(gt=0, allow_inf_nan=False)


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hand: str
    task: str
    method: Literal["iid", "eigendexplore"]
    seed: int = Field(ge=0, le=2**32 - 1)
    basis: Literal["human", "random"] | None = None
    retained_variance: float | None = Field(
        default=None, gt=0, le=1, allow_inf_nan=False
    )
    k: int | None = Field(default=None, gt=0)
    iid_scale: float = Field(ge=0, allow_inf_nan=False)
    eigen_scale: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_recipe(self):
        if self.hand not in defaults()["hands"]:
            raise ValueError("unsupported Spider hand")
        if self.task not in task_names():
            raise ValueError("unknown task")
        if self.method == "iid":
            if (
                self.basis,
                self.retained_variance,
                self.k,
                self.iid_scale,
                self.eigen_scale,
            ) != (None, None, None, 1.0, 0.0):
                raise ValueError("IID baseline uses stock noise only")
        elif (
            self.basis is None
            or self.retained_variance is None
            or self.eigen_scale <= 0
        ):
            raise ValueError(
                "EigenDExplore requires a basis, a retention, and a positive Eigen scale"
            )
        elif self.k != rank(self.hand, self.retained_variance):
            raise ValueError("rank differs from the basis spectrum")
        return self

    @property
    def setting_id(self):
        if self.method == "iid":
            return "iid"
        return f"eigendexplore-{self.basis}-r{self.retained_variance:g}-k{self.k}-i{self.iid_scale:g}-e{self.eigen_scale:g}"

    @property
    def id(self):
        return f"{self.hand}/{self.setting_id}/{self.task}/s{self.seed}"


def settings_grid(values=None):
    """Every combination of the given values; unspecified names use the preset."""
    values = dict(values or {})
    unknown = set(values) - set(SETTING_NAMES)
    if unknown:
        raise ValueError(f"unknown Spider settings: {sorted(unknown)}")
    axes = [values.get(name, [defaults()[name]]) for name in SETTING_NAMES]
    for name, axis in zip(SETTING_NAMES, axes):
        if not axis or len(axis) != len(set(axis)):
            raise ValueError(f"{name} needs distinct values")
    return [
        Setting(**dict(zip(SETTING_NAMES, combination)))
        for combination in itertools.product(*axes)
    ]


def make_jobs(hand, *, tasks=None, seeds=None, settings=None):
    settings = settings_grid() if settings is None else list(settings)
    tasks = list(tasks) if tasks else list(task_names())
    seeds = list(seeds) if seeds else defaults()["seeds"]
    for values, label in [(tasks, "tasks"), (seeds, "seeds"), (settings, "settings")]:
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label}")
    jobs = []
    for task in tasks:
        for seed in seeds:
            jobs.append(
                Job(
                    hand=hand,
                    task=task,
                    seed=seed,
                    method="iid",
                    iid_scale=1.0,
                    eigen_scale=0.0,
                )
            )
            for setting in settings:
                jobs.append(
                    Job(
                        hand=hand,
                        task=task,
                        seed=seed,
                        method="eigendexplore",
                        k=rank(hand, setting.retained_variance),
                        **setting.model_dump(),
                    )
                )
    return jobs
