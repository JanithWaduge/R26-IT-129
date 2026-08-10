import json
from pathlib import Path

import pandas as pd

from app.ml.recall_predictor import (
    RecallModelPredictor,
)


DATASET_PATH = Path(
    "data/synthetic/"
    "receptive_recall_training_synthetic.csv"
)

MODEL_PATH = Path(
    "outputs/models/receptive_recall/"
    "recall_model_champion.joblib"
)

METADATA_PATH = Path(
    "outputs/models/receptive_recall/"
    "model_metadata.json"
)


def main() -> None:
    predictor = RecallModelPredictor(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
    )

    dataset = pd.read_csv(
        DATASET_PATH
    )

    feature_row = (
        dataset.iloc[0].to_dict()
    )

    probability = (
        predictor.predict_probability(
            feature_row
        )
    )

    print(
        json.dumps(
            {
                "future_recall_probability": (
                    round(
                        probability,
                        6,
                    )
                ),
                "predicted_success": (
                    probability >= 0.50
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()