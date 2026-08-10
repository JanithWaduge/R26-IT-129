import json

import joblib
import numpy as np

from app.core.config import Settings
from app.ml.model_runtime import RecallModelRuntime


class DummyProbabilityModel:
    def predict_proba(self, frame):
        del frame
        return np.array([[0.25, 0.75]])


def settings(**updates) -> Settings:
    values = {
        "jwt_access_secret": "a" * 128,
        "jwt_refresh_secret": "b" * 128,
        "ml_scheduler_enabled": False,
    }
    values.update(updates)
    return Settings(**values)


def test_disabled_runtime_uses_fallback() -> None:
    status = RecallModelRuntime(settings()).status()
    assert status.status == "disabled"
    assert status.probability is None


def test_model_file_missing(tmp_path) -> None:
    status = RecallModelRuntime(settings(
        ml_scheduler_enabled=True,
        ml_model_path=str(tmp_path / "missing.joblib"),
        ml_model_metadata_path=str(tmp_path / "missing.json"),
    )).status()
    assert status.status == "unavailable"
    assert "Model file not found" in status.message


def write_runtime_files(tmp_path, *, synthetic=False, version=None):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(DummyProbabilityModel(), model_path)
    metadata_path.write_text(json.dumps({
        "model_name": "test-model", "model_version": "test-1",
        "training_data_is_synthetic": synthetic,
        "scikit_learn_version": version,
        "all_model_features": ["anchor_correct"],
    }), encoding="utf-8")
    return model_path, metadata_path


def test_synthetic_model_is_blocked(tmp_path) -> None:
    model, metadata = write_runtime_files(tmp_path, synthetic=True)
    status = RecallModelRuntime(settings(
        ml_scheduler_enabled=True, ml_model_path=str(model),
        ml_model_metadata_path=str(metadata), ml_allow_synthetic_model=False,
    )).status()
    assert status.status == "unavailable"
    assert "Synthetic" in status.message


def test_version_mismatch_is_blocked(tmp_path) -> None:
    model, metadata = write_runtime_files(tmp_path, version="0.0.invalid")
    status = RecallModelRuntime(settings(
        ml_scheduler_enabled=True, ml_model_path=str(model),
        ml_model_metadata_path=str(metadata), ml_require_exact_sklearn_version=True,
    )).status()
    assert status.status == "unavailable"
    assert "version mismatch" in status.message


def test_missing_features_and_probability_bounds(tmp_path) -> None:
    model, metadata = write_runtime_files(tmp_path)
    runtime = RecallModelRuntime(settings(
        ml_scheduler_enabled=True, ml_model_path=str(model),
        ml_model_metadata_path=str(metadata), ml_require_exact_sklearn_version=False,
    ))
    assert runtime.predict({}).status == "invalid_features"
    prediction = runtime.predict({"anchor_correct": 1})
    assert prediction.status == "predicted"
    assert 0.0 <= prediction.probability <= 1.0
