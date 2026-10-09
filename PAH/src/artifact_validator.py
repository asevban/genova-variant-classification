from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(root: Path, manifest_path: Path | None = None) -> list[str]:
    manifest_path = manifest_path or root / "config" / "artifact_manifest.json"
    manifest = load_json(manifest_path)
    issues: list[str] = []
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if not path.exists():
            issues.append(f"MISSING artifact: {item['path']}")
            continue
        size = path.stat().st_size
        if int(item["size_bytes"]) != size:
            issues.append(f"SIZE mismatch: {item['path']}")
        current = sha256(path)
        if current != item["sha256"]:
            issues.append(f"SHA256 mismatch: {item['path']}")
    return issues
