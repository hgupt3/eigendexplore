"""Strict manifest loading."""

from pathlib import Path

import yaml
from pydantic import BaseModel


def read_manifest(path: Path, model: type[BaseModel]) -> BaseModel:
    if not path.is_file():
        raise FileNotFoundError(path)
    return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
