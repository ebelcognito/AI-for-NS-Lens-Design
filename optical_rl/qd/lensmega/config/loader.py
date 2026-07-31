from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .schema import RunConfig


def load_config(path: Union[str, Path]) -> RunConfig:
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
