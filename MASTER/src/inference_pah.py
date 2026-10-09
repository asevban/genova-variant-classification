from __future__ import annotations

from pathlib import Path
from typing import List

from .inference_common import BasePanelAdapter


class PAHAdapter(BasePanelAdapter):
    panel = "PAH"

    def command(self, input_csv: Path, output_csv: Path) -> List[str]:
        panel_dir = self.panel_dir()
        return [
            str(panel_dir / "final" / "predict.py"),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--bundle",
            str(panel_dir / "final" / "models" / "final_model_bundle_v2.pkl"),
        ]
