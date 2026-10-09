from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.input_loader import PANEL_NAMES, normalize_frame, read_table
from src.json_writer import build_submission, write_json
from src.logging_utils import setup_logging
from src.panel_router import get_adapter
from src.submission_validator import validate_payload


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GENOVA panel-based final JSON runner")
    parser.add_argument("--panel", required=True, choices=PANEL_NAMES, help="Calistirilacak panel.")
    parser.add_argument("--input", required=True, help="Bu panele ait test CSV/XLSX yolu.")
    parser.add_argument("--output", default="", help="Opsiyonel panel JSON output path.")
    parser.add_argument("--team-id", default="", help="team.json icindeki team_id degerini gecici override eder.")
    parser.add_argument("--application-id", default="", help="team.json icindeki application_id degerini gecici override eder.")
    return parser.parse_args()


def require_team_ready(team: Dict[str, str]) -> None:
    missing = [key for key in ("team_name", "team_id", "application_id", "competition_level") if not team.get(key)]
    if missing:
        raise RuntimeError("Team config eksik: " + ", ".join(missing))
    if team["competition_level"] != "UNIVERSITE_VE_UZERI":
        raise RuntimeError("competition_level UNIVERSITE_VE_UZERI olmali.")


def panel_output_name(team_id: str, panel: str) -> str:
    clean = team_id.strip() or "MISSING_TEAM_ID"
    return f"TEAM_{clean}_{panel}_FINAL.json"


def without_json_suffix(name: str) -> str:
    if name.endswith(".json"):
        return name[:-5]
    return name


def resolve_panel_root(root: Path, panel: str) -> Path:
    if root.name.upper() == panel and (root / "final").exists():
        return root
    nested = root / "panels" / panel
    if nested.exists():
        return nested
    return root


def main() -> int:
    args = parse_args()
    panel = args.panel.upper()
    registry = load_json(ROOT / "config" / "panel_registry.json")
    class_mapping = load_json(ROOT / "config" / "class_mapping.json")
    team = load_json(ROOT / "config" / "team.json")
    if args.team_id:
        team["team_id"] = args.team_id
    if args.application_id:
        team["application_id"] = args.application_id
    require_team_ready(team)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"{panel} input dosyasi bulunamadi: {input_path}")

    frame = normalize_frame(read_table(input_path), panel=panel)
    expected_ids = frame["Variant_ID"].astype(str).tolist()
    panel_root = resolve_panel_root(ROOT, panel)
    panel_output = panel_root / "output"
    panel_output.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(ROOT)
    adapter = get_adapter(panel, ROOT, registry)
    adapter.load()
    adapter.work_dir = panel_output / "_work"
    adapter.work_dir.mkdir(parents=True, exist_ok=True)

    python = adapter.panel_python()
    entrypoint = ROOT / registry[panel]["entrypoint"]
    help_check = subprocess.run(
        [str(python), str(entrypoint), "--help"],
        cwd=str(ROOT / registry[panel]["runtime_dir"]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    if help_check.returncode != 0:
        raise RuntimeError(f"{panel} --help failed:\n{help_check.stdout[-1000:]}")

    logging.info("Panel final run started: %s rows=%s", panel, len(frame))
    result = adapter.predict(frame)
    payload = build_submission(team, result.rows)
    ok, validation_text, issues = validate_payload(payload, expected_ids, class_mapping, registry)

    stem = without_json_suffix(panel_output_name(team["team_id"], panel))
    validation_path = panel_output / f"{stem}.validation.txt"
    summary_path = panel_output / f"{stem}.run_summary.json"
    validation_path.write_text(validation_text, encoding="utf-8")

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "panel": panel,
        "input": str(input_path),
        "input_row_count": len(expected_ids),
        "output_row_count": len(result.rows),
        "model_artifacts": [str(path.relative_to(ROOT)) for path in result.artifact_paths],
        "threshold": result.threshold,
        "validation_status": "PASS" if ok else "FAIL",
        "validation_issues": issues,
        "log_file": str(log_path.relative_to(ROOT)),
    }
    write_json(summary_path, summary)

    if not ok:
        print(validation_text)
        print(f"{panel} FINAL JSON olusturulmadi; validator FAIL.")
        return 2

    output_path = Path(args.output).resolve() if args.output else panel_output / panel_output_name(team["team_id"], panel)
    write_json(output_path, payload)
    print(validation_text)
    print("================================")
    print(f"GENOVA {panel} PANEL JSON READY")
    print("================================")
    print(f"Panel: {panel}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Validation: PASS")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
