"""Species configuration loader."""
from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SpeciesConfig:
    """Consolidated optical + retinal parameters for one species."""
    name: str
    optical: object
    retinal: object

    @classmethod
    def load(cls, species_name: str) -> "SpeciesConfig":
        """Load species config from data/species/{species_name}.yaml."""
        raise NotImplementedError
