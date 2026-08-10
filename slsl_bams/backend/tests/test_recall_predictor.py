import json
from pathlib import Path

from app.ml.recall_predictor import RecallModelPredictor


def test_champion_predicts_probability() -> None:
    package_root = Path(__file__).resolve().parents[1]

    predictor = RecallModelPredictor(
        model_path=(
            package_root
            / "outputs/models/receptive_recall/"
            "recall_model_champion.joblib"
        ),
        metadata_path=(
            package_root
            / "outputs/models/receptive_recall/"
            "model_metadata.json"
        ),
    )

    metadata = json.loads(
        (package_root / "outputs/models/receptive_recall/model_metadata.json")
        .read_text(encoding="utf-8")
    )
    categorical = set(metadata["categorical_features"])
    row = {
        feature: ("unknown" if feature in categorical else 0.0)
        for feature in metadata["all_model_features"]
    }

    probability = (
        predictor.predict_probability(
            row
        )
    )

    assert 0.0 <= probability <= 1.0
