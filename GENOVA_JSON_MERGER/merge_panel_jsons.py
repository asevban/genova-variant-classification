from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_PANELS = ("CFTR", "KANSER", "MASTER", "PAH")
ALLOWED_CLASSES = {"0", "1"}
TEAM_DEFAULTS = {
    "team_name": "GENOVA",
    "team_id": "885171",
    "application_id": "4807248",
    "competition_level": "UNIVERSITE_VE_UZERI",
}


class MergeError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MergeError(f"JSON okunamadi: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise MergeError(f"JSON kok objesi object olmali: {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pick(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def normalize_class(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if int(value) in (0, 1) and float(value) == int(value):
            return str(int(value))
    text = str(value).strip().upper()
    if text in {"0", "BENIGN", "B", "NEGATIVE", "NEG", "FALSE"}:
        return "0"
    if text in {"1", "PATHOGENIC", "P", "POSITIVE", "POS", "TRUE"}:
        return "1"
    raise MergeError(f"Gecersiz class/label degeri: {value!r}")


def normalize_probability(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError) as exc:
        raise MergeError(f"Gecersiz probability degeri: {value!r}") from exc
    if not math.isfinite(prob) or prob < 0 or prob > 1:
        raise MergeError(f"Probability 0-1 araliginda olmali: {value!r}")
    return prob


def infer_panel(path: Path, payload: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    candidates = [
        pick(payload, ("panel", "Panel", "panel_name", "panelName")),
        pick(row or {}, ("panel", "Panel", "panel_name", "panelName")),
        path.stem,
        path.name,
    ]
    for candidate in candidates:
        text = str(candidate or "").upper()
        for panel in REQUIRED_PANELS:
            if panel in text:
                return panel
    raise MergeError(f"Panel adi anlasilamadi: {path}")


def prediction_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    predictions = payload.get("predictions")
    if predictions is None:
        predictions = payload.get("rows")
    if predictions is None and "Variant_ID" in payload:
        predictions = [payload]
    if not isinstance(predictions, list):
        raise MergeError(f"predictions listesi yok veya liste degil: {path}")
    if not predictions:
        raise MergeError(f"predictions listesi bos: {path}")
    if not all(isinstance(row, dict) for row in predictions):
        raise MergeError(f"predictions icindeki tum satirlar object olmali: {path}")
    return predictions


def normalize_prediction(path: Path, payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    panel = infer_panel(path, payload, row)
    variant_id = pick(row, ("id", "Variant_ID", "variant_id", "ID"))
    if variant_id in (None, ""):
        raise MergeError(f"{path}: Variant_ID/id eksik")
    predicted_class = normalize_class(
        pick(row, ("predicted_class", "Label", "label", "predicted_label", "prediction", "prediction_name", "class"))
    )
    predicted_prob = normalize_probability(
        pick(row, ("predicted_prob", "predicted_probability", "pathogenic_score", "score_pathogenic", "probability", "prob"))
    )
    normalized = {
        "id": str(variant_id).strip(),
        "panel": panel,
        "predicted_class": predicted_class,
    }
    if predicted_prob is not None:
        normalized["predicted_prob"] = round(predicted_prob, 10)
    return normalized


def discover_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise MergeError(f"Input klasoru bulunamadi: {input_dir}")
    files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not files:
        raise MergeError(
            "Input klasorunde panel JSON bulunamadi. Panel ciktilarini GENOVA_JSON_MERGER/input icine kopyalayin."
        )
    return files


def team_from_payloads(payloads: list[dict[str, Any]]) -> dict[str, str]:
    team = dict(TEAM_DEFAULTS)
    seen: dict[str, set[str]] = defaultdict(set)
    for payload in payloads:
        for field in TEAM_DEFAULTS:
            value = payload.get(field)
            if value not in (None, ""):
                seen[field].add(str(value))
    for field, values in seen.items():
        if len(values) > 1:
            raise MergeError(f"Panel JSON takim bilgileri uyusmuyor: {field}={sorted(values)}")
        if values:
            team[field] = next(iter(values))
    return team


def validate_predictions(predictions: list[dict[str, Any]], required_panels: bool) -> list[str]:
    issues: list[str] = []
    panel_id_pairs = [(row["panel"], row["id"]) for row in predictions]
    pair_counts = Counter(panel_id_pairs)
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count > 1)
    if duplicate_pairs:
        examples = ", ".join(f"{panel}:{variant_id}" for panel, variant_id in duplicate_pairs[:10])
        issues.append("Ayni panel icinde duplicate Variant_ID/id var: " + examples)

    panels = Counter(row["panel"] for row in predictions)
    if required_panels:
        missing = [panel for panel in REQUIRED_PANELS if panels[panel] == 0]
        if missing:
            issues.append("Eksik panel JSON: " + ", ".join(missing))

    for index, row in enumerate(predictions):
        if not row["id"].strip():
            issues.append(f"Satir {index}: bos id")
        if row["panel"] not in REQUIRED_PANELS:
            issues.append(f"Satir {index}: gecersiz panel {row['panel']!r}")
        if row["predicted_class"] not in ALLOWED_CLASSES:
            issues.append(f"Satir {index}: gecersiz class {row['predicted_class']!r}")
        if "predicted_prob" in row:
            prob = row["predicted_prob"]
            if not isinstance(prob, (int, float)) or not math.isfinite(float(prob)) or not 0 <= float(prob) <= 1:
                issues.append(f"Satir {index}: gecersiz predicted_prob {prob!r}")
    return issues


def build_internal_payload(team: dict[str, str], predictions: list[dict[str, Any]], source_files: list[Path]) -> dict[str, Any]:
    panel_counts = Counter(row["panel"] for row in predictions)
    return {
        **team,
        "submission_format": "GENOVA_INTERNAL_MERGED_JSON_V1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": [path.name for path in source_files],
        "panel_counts": {panel: panel_counts.get(panel, 0) for panel in REQUIRED_PANELS},
        "total_predictions": len(predictions),
        "predictions": predictions,
    }


def build_official_payload(team: dict[str, str], predictions: list[dict[str, Any]], source_files: list[Path]) -> dict[str, Any]:
    return {
        **team,
        "submission_format": "OFFICIAL_CANDIDATE_JSON_MINIMAL_VERIFY_WITH_GUIDE",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": [path.name for path in source_files],
        "predictions": [
            {
                "Variant_ID": row["id"],
                "Label": row["predicted_class"],
            }
            for row in predictions
        ],
    }


def merge(input_dir: Path, output_path: Path, fmt: str, allow_missing_panels: bool) -> tuple[dict[str, Any], str]:
    files = discover_json_files(input_dir)
    payloads = [read_json(path) for path in files]
    team = team_from_payloads(payloads)

    predictions: list[dict[str, Any]] = []
    for path, payload in zip(files, payloads):
        for row in prediction_rows(payload, path):
            predictions.append(normalize_prediction(path, payload, row))

    predictions.sort(key=lambda row: (REQUIRED_PANELS.index(row["panel"]), row["id"]))
    issues = validate_predictions(predictions, required_panels=not allow_missing_panels)
    if issues:
        raise MergeError("Merge validation FAIL:\n- " + "\n- ".join(issues))

    if fmt == "official":
        output = build_official_payload(team, predictions, files)
    else:
        output = build_internal_payload(team, predictions, files)
    write_json(output_path, output)

    panel_counts = Counter(row["panel"] for row in predictions)
    summary_lines = [
        "==========================",
        "GENOVA JSON MERGE VALIDATION",
        "==========================",
        f"Input files: {len(files)}",
        f"Total predictions: {len(predictions)}",
        f"Output format: {fmt}",
        f"Output: {output_path}",
        "",
    ]
    summary_lines.extend(f"{panel}: {panel_counts.get(panel, 0)} rows" for panel in REQUIRED_PANELS)
    summary_lines.extend(["", "STATUS: PASS"])
    return output, "\n".join(summary_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GENOVA panel JSON birlestirici")
    parser.add_argument("--input-dir", default="input", help="Panel JSON dosyalarinin bulundugu klasor.")
    parser.add_argument("--output", default="output/GENOVA_FINAL_SUBMISSION.json", help="Birlesik JSON cikti yolu.")
    parser.add_argument("--format", choices=("internal", "official"), default="internal", help="Cikti semasi.")
    parser.add_argument("--allow-missing-panels", action="store_true", help="Eksik panel varsa FAIL verme; sadece prova icin.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_dir = (root / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    try:
        _, summary = merge(input_dir, output_path, args.format, args.allow_missing_panels)
    except MergeError as exc:
        print("==========================")
        print("GENOVA JSON MERGE VALIDATION")
        print("==========================")
        print("STATUS: FAIL")
        print(str(exc))
        return 2
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
