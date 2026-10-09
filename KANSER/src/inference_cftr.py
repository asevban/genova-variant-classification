from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .inference_common import BasePanelAdapter


class CFTRAdapter(BasePanelAdapter):
    panel = "CFTR"

    def required_paths(self) -> List[str]:
        base = "final/models/artifacts" if self.panel_dir() == self.root else "panels/CFTR/final/models/artifacts"
        paths = [f"{base}/preprocessing.json"]
        paths.extend(f"{base}/id3_repeat_{idx}.json" for idx in range(1, 6))
        paths.extend(f"{base}/catboost_repeat_{idx}.cbm" for idx in range(1, 6))
        paths.extend(f"{base}/randomforest_repeat_{idx}.joblib" for idx in range(1, 6))
        return paths

    def command(self, input_csv: Path, output_csv: Path) -> List[str]:
        panel_dir = self.panel_dir()
        self.audit_csv = self.work_dir / "CFTR_audit.csv"
        self.drift_json = self.work_dir / "CFTR_drift.json"
        return [
            str(panel_dir / "final" / "predict.py"),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--mode",
            "main",
            "--audit",
            str(self.audit_csv),
            "--drift-report",
            str(self.drift_json),
        ]

    def parse_output(self, output_csv: Path) -> pd.DataFrame:
        output = pd.read_csv(output_csv)
        audit = pd.read_csv(self.audit_csv)
        return output.merge(audit[["Variant_ID", "probability"]], on="Variant_ID", how="left")
