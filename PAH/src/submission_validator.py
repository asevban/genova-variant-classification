from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


ALLOWED_PANELS = {"MASTER", "KANSER", "PAH", "CFTR"}
ALLOWED_CLASSES = {"0", "1"}


def validate_payload(
    payload: Dict[str, Any],
    expected_ids: Iterable[str],
    class_mapping: Dict[str, Any],
    registry: Dict[str, Any],
) -> tuple[bool, str, list[str]]:
    issues: list[str] = []
    expected = [str(x) for x in expected_ids]
    expected_set = set(expected)

    for field in ("team_name", "team_id", "application_id", "competition_level", "predictions"):
        if field not in payload or payload[field] in ("", None):
            issues.append(f"Missing team-level field: {field}")
    if payload.get("competition_level") != "UNIVERSITE_VE_UZERI":
        issues.append("competition_level must be UNIVERSITE_VE_UZERI")
    if class_mapping.get("status") != "CONFIRMED":
        issues.append("Class mapping is not CONFIRMED")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        predictions = []
        issues.append("predictions must be an array")

    produced_ids: list[str] = []
    invalid_panels = invalid_classes = invalid_probabilities = nan_inf = threshold_mismatch = 0
    for idx, row in enumerate(predictions):
        if not isinstance(row, dict):
            issues.append(f"Prediction row {idx} is not an object")
            continue
        for field in ("id", "panel", "predicted_class", "predicted_prob"):
            if field not in row:
                issues.append(f"Row {idx} missing field: {field}")
        row_id = str(row.get("id", ""))
        produced_ids.append(row_id)
        panel = str(row.get("panel", "")).upper()
        if panel not in ALLOWED_PANELS:
            invalid_panels += 1
        predicted_class = str(row.get("predicted_class", ""))
        if predicted_class not in ALLOWED_CLASSES:
            invalid_classes += 1
        prob = row.get("predicted_prob")
        if not isinstance(prob, (int, float)):
            invalid_probabilities += 1
            continue
        if not math.isfinite(float(prob)):
            nan_inf += 1
            continue
        if float(prob) < 0 or float(prob) > 1:
            invalid_probabilities += 1
        threshold = registry.get(panel, {}).get("threshold")
        if threshold is None:
            issues.append(f"{panel} threshold is null")
        else:
            expected_class = "1" if float(prob) >= float(threshold) else "0"
            if predicted_class in ALLOWED_CLASSES and predicted_class != expected_class:
                threshold_mismatch += 1

    counts = Counter(produced_ids)
    duplicate_ids = sum(1 for count in counts.values() if count > 1)
    missing = sorted(expected_set - set(produced_ids))
    extra = sorted(set(produced_ids) - expected_set)

    if duplicate_ids:
        issues.append(f"Duplicate IDs: {duplicate_ids}")
    if missing:
        issues.append(f"Missing IDs: {len(missing)}")
    if extra:
        issues.append(f"Extra IDs: {len(extra)}")
    if invalid_panels:
        issues.append(f"Invalid panels: {invalid_panels}")
    if invalid_classes:
        issues.append(f"Invalid classes: {invalid_classes}")
    if invalid_probabilities:
        issues.append(f"Invalid probabilities: {invalid_probabilities}")
    if nan_inf:
        issues.append(f"NaN/Inf probabilities: {nan_inf}")
    if threshold_mismatch:
        issues.append(f"Class/threshold mismatches: {threshold_mismatch}")

    lines = [
        "==========================",
        "FINAL SUBMISSION VALIDATION",
        "==========================",
        f"Rows expected: {len(expected)}",
        f"Rows produced: {len(predictions)}",
        "",
        f"Duplicate IDs: {duplicate_ids}",
        f"Missing IDs: {len(missing)}",
        f"Extra IDs: {len(extra)}",
        f"Invalid panels: {invalid_panels}",
        f"Invalid classes: {invalid_classes}",
        f"Invalid probabilities: {invalid_probabilities}",
        f"NaN/Inf: {nan_inf}",
        f"Class/threshold mismatches: {threshold_mismatch}",
        "",
        f"STATUS: {'PASS' if not issues else 'FAIL'}",
    ]
    if issues:
        lines.extend(["", "Issues:"])
        lines.extend(f"- {issue}" for issue in issues)
    return not issues, "\n".join(lines) + "\n", issues


def validate_file(
    json_path: Path,
    expected_ids_path: Path,
    class_mapping_path: Path,
    registry_path: Path,
) -> tuple[bool, str, list[str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    expected_ids = json.loads(expected_ids_path.read_text(encoding="utf-8"))["expected_ids"]
    class_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return validate_payload(payload, expected_ids, class_mapping, registry)
