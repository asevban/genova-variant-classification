from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TARGET_COLUMNS = {"Label", "label", "TARGET", "target", "y"}
ROUTING_COLUMNS = {"panel", "id", "ID", "Variant_ID", "variant_id"}


@dataclass
class PredictionResult:
    panel: str
    rows: List[Dict[str, Any]]
    output_csv: Path
    artifact_paths: List[Path]
    threshold: float


class BasePanelAdapter:
    panel: str = ""

    def __init__(self, root: Path, registry: Dict[str, Any]):
        self.root = root
        self.registry = registry
        self.panel_config = registry[self.panel]
        self.runtime_dir = root / self.panel_config["runtime_dir"]
        self.work_dir = root / "output" / "_panel_work" / self.panel
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        for rel in self.required_paths():
            path = self.root / rel
            if not path.exists():
                raise FileNotFoundError(f"{self.panel} required artifact missing: {rel}")

    def required_paths(self) -> List[str]:
        model_path = self.panel_config["model_path"]
        if model_path.endswith("*"):
            return []
        return [model_path]

    def panel_dir(self) -> Path:
        nested = self.root / "panels" / self.panel
        if nested.exists():
            return nested
        return self.root

    def panel_python(self) -> Path:
        candidates = [
            self.root / ".venv" / "Scripts" / "python.exe",
            self.root / ".venv" / "bin" / "python",
            self.root / ".venvs" / self.panel.lower() / "Scripts" / "python.exe",
            self.root / ".venvs" / self.panel.lower() / "bin" / "python",
            self.root / "panels" / self.panel / ".venv" / "Scripts" / "python.exe",
            self.root / "panels" / self.panel / ".venv" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{self.panel} Python ortami bulunamadi. Yarismadan once .venvs/{self.panel.lower()} "
            f"veya panels/{self.panel}/.venv klasoru bu isletim sistemi icin hazir olmali."
        )

    def prepare_input_csv(self, frame: pd.DataFrame) -> Path:
        if "Variant_ID" not in frame.columns:
            raise ValueError(f"{self.panel}: Variant_ID kolonu yok.")
        out = frame.copy()
        drop_cols = [c for c in out.columns if c in TARGET_COLUMNS or c in ROUTING_COLUMNS]
        out = out.drop(columns=drop_cols, errors="ignore")
        out.insert(0, "Variant_ID", frame["Variant_ID"].astype(str).to_numpy())
        path = self.work_dir / f"{self.panel}_input.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def validate_input(self, frame: pd.DataFrame) -> None:
        if "Variant_ID" not in frame.columns:
            raise ValueError(f"{self.panel}: Variant_ID zorunlu.")
        if frame["Variant_ID"].duplicated().any():
            raise ValueError(f"{self.panel}: duplicate Variant_ID var.")

    def command(self, input_csv: Path, output_csv: Path) -> List[str]:
        raise NotImplementedError

    def parse_output(self, output_csv: Path) -> pd.DataFrame:
        return pd.read_csv(output_csv)

    def output_to_rows(self, output: pd.DataFrame) -> List[Dict[str, Any]]:
        label_col = self.panel_config["output_label_column"]
        prob_col = self.panel_config["output_probability_column"]
        for column in ("Variant_ID", label_col, prob_col):
            if column and column not in output.columns:
                raise RuntimeError(f"{self.panel}: output missing {column}")
        return [
            {
                "id": str(row["Variant_ID"]),
                "panel": self.panel,
                "predicted_class": str(int(row[label_col])),
                "predicted_prob": float(row[prob_col]),
            }
            for _, row in output.iterrows()
        ]

    def predict(self, frame: pd.DataFrame) -> PredictionResult:
        self.load()
        self.validate_input(frame)
        input_csv = self.prepare_input_csv(frame)
        output_csv = self.work_dir / f"{self.panel}_predictions.csv"
        cmd = [str(self.panel_python()), *self.command(input_csv, output_csv)]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        subprocess.run(cmd, cwd=str(self.runtime_dir), check=True, env=env)
        output = self.parse_output(output_csv)
        rows = self.output_to_rows(output)
        return PredictionResult(
            panel=self.panel,
            rows=rows,
            output_csv=output_csv,
            artifact_paths=[self.root / self.panel_config["model_path"]],
            threshold=float(self.panel_config["threshold"]),
        )


def write_expected_ids(path: Path, ids: List[str]) -> None:
    path.write_text(json.dumps({"expected_ids": ids}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
