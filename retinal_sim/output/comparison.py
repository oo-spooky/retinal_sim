"""Multi-species comparison panel renderer."""
from __future__ import annotations

from typing import Dict


def render_comparison(activations: Dict[str, object], **kwargs) -> object:
    """Render a side-by-side panel: input | human | dog | cat."""
    raise NotImplementedError
