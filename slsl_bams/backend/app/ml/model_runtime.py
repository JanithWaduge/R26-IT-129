from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from app.core.config import Settings


@dataclass(frozen=True)
class RuntimePrediction:
    status: str
    probability: float | None
    model_name: str | None
    model_version: str | None
    synthetic_model: bool
    message: str | None


class RecallModelRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._metadata: dict | None = None
        self._load_error: str | None = None

    @property
    def metadata(self) -> dict | None:
        return self._metadata

    def _load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        if not self.settings.ml_scheduler_enabled:
            self._load_error = "ML scheduling is disabled."
            return
        model_path = Path(self.settings.ml_model_path)
        metadata_path = Path(self.settings.ml_model_metadata_path)
        if not model_path.exists():
            self._load_error = f"Model file not found: {model_path}"
            return
        if not metadata_path.exists():
            self._load_error = f"Metadata file not found: {metadata_path}"
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self._load_error = f"Model metadata is invalid: {error}"
            return
        self._metadata = metadata
        synthetic = bool(metadata.get("training_data_is_synthetic", False))
        if synthetic and not self.settings.ml_allow_synthetic_model:
            self._load_error = "Synthetic recall model is blocked by configuration."
            return
        trained = metadata.get("scikit_learn_version")
        if (self.settings.ml_require_exact_sklearn_version and trained
                and trained != sklearn.__version__):
            self._load_error = (
                f"scikit-learn version mismatch. Model={trained}; Runtime={sklearn.__version__}."
            )
            return
        try:
            self._model = joblib.load(model_path)
        except Exception as error:
            self._load_error = f"Model could not be loaded: {error}"
            return
        features = metadata.get("all_model_features")
        if not isinstance(features, list) or not features:
            self._model = None
            self._load_error = "Model metadata does not contain a valid feature list."

    def status(self) -> RuntimePrediction:
        self._load()
        metadata = self._metadata or {}
        if self._model is None:
            return RuntimePrediction(
                "disabled" if not self.settings.ml_scheduler_enabled else "unavailable",
                None, metadata.get("model_name"), metadata.get("model_version"),
                bool(metadata.get("training_data_is_synthetic", False)), self._load_error,
            )
        return RuntimePrediction(
            "ready", None, metadata.get("model_name"), metadata.get("model_version"),
            bool(metadata.get("training_data_is_synthetic", False)), None,
        )

    def predict(self, features: dict[str, Any]) -> RuntimePrediction:
        status = self.status()
        if status.status != "ready":
            return status
        assert self._model is not None and self._metadata is not None
        required = list(self._metadata["all_model_features"])
        missing = [feature for feature in required if feature not in features]
        if missing:
            return RuntimePrediction(
                "invalid_features", None, status.model_name, status.model_version,
                status.synthetic_model, "Missing model features: " + ", ".join(missing),
            )
        try:
            probability = float(self._model.predict_proba(
                pd.DataFrame([{key: features[key] for key in required}])
            )[0, 1])
        except Exception as error:
            return RuntimePrediction(
                "prediction_failed", None, status.model_name, status.model_version,
                status.synthetic_model, str(error),
            )
        return RuntimePrediction(
            "predicted", max(0.0, min(1.0, probability)), status.model_name,
            status.model_version, status.synthetic_model, None,
        )
