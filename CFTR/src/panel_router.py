from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .inference_cftr import CFTRAdapter
from .inference_common import BasePanelAdapter
from .inference_kanser import KANSERAdapter
from .inference_master import MASTERAdapter
from .inference_pah import PAHAdapter


ADAPTERS = {
    "MASTER": MASTERAdapter,
    "KANSER": KANSERAdapter,
    "PAH": PAHAdapter,
    "CFTR": CFTRAdapter,
}


def get_adapter(panel: str, root: Path, registry: Dict[str, Any]) -> BasePanelAdapter:
    if panel not in ADAPTERS:
        raise ValueError(f"UNKNOWN PANEL: {panel}")
    return ADAPTERS[panel](root=root, registry=registry)
