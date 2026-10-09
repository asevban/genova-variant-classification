from __future__ import annotations

from pathlib import Path
from typing import List

from .inference_common import BasePanelAdapter


class KANSERAdapter(BasePanelAdapter):
    panel = "KANSER"

    def command(self, input_csv: Path, output_csv: Path) -> List[str]:
        panel_dir = self.panel_dir()
        return [
            str(panel_dir / "final" / "predict.py"),
            "--model",
            str(panel_dir / "final" / "models" / "KANSER_FINAL_BENIGN_SHIFT_MODEL.joblib"),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
        ]
