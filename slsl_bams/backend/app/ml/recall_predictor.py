from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class RecallModelPredictor:
    def __init__(
        self,
        *,
        model_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)

        # Only load model files created by your own trusted training pipeline.
        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )
        self.feature_columns = list(
            self.metadata["all_model_features"]
        )

    def predict_probability(
        self,
        features: dict[str, Any],
    ) -> float:
        missing = [
            column
            for column in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                "Missing recall-model features: "
                + ", ".join(missing)
            )

        frame = pd.DataFrame(
            [
                {
                    column: features[column]
                    for column in self.feature_columns
                }
            ]
        )

        probability = self.model.predict_proba(frame)[0, 1]

        return float(probability)
