from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def output_name(team_id: str) -> str:
    clean = team_id.strip() or "MISSING_TEAM_ID"
    return f"TEAM_{clean}_FINAL.json"


def build_submission(team: Dict[str, str], predictions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "team_name": team.get("team_name", ""),
        "team_id": team.get("team_id", ""),
        "application_id": team.get("application_id", ""),
        "competition_level": team.get("competition_level", ""),
        "predictions": list(predictions),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
