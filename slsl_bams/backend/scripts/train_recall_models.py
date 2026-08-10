from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "category_code",
    "prompt_language",
    "selection_strategy",
    "selection_reason",
    "anchor_outcome",
]

NUMERIC_FEATURES = [
    "anchor_correct",
    "anchor_was_skipped",
    "anchor_quality_score",
    "anchor_quality_band",
    "response_time_ms",
    "retry_count",
    "hint_count",
    "sign_difficulty",
    "was_due",
    "days_overdue",
    "selection_priority",
    "pre_mastery_score",
    "post_mastery_score",
    "pre_total_reviews",
    "pre_successful_reviews",
    "pre_failure_count",
    "pre_average_quality_score",
    "pre_average_response_time_ms",
    "hours_since_previous_review",
    "pre_ease_factor",
    "post_ease_factor",
    "pre_interval_days",
    "post_interval_days",
    "pre_repetition_count",
    "post_repetition_count",
    "pre_lapse_count",
    "post_lapse_count",
    "scheduled_delay_hours",
    "learner_prior_attempts",
    "learner_prior_accuracy",
    "learner_prior_average_quality",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "future_recall_success"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/models/receptive_recall",
    )
    return parser.parse_args()


def calibration_bucket(learner_id: str) -> int:
    digest = hashlib.sha256(
        f"calibration:{learner_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) % 100


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:
    predicted = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predicted,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predicted,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                predicted,
            ).tolist()
        ),
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(dataset_path)

    required = {
        "row_id",
        "learner_group_id",
        "split",
        TARGET,
        *FEATURES,
    }

    missing = sorted(
        required - set(data.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing dataset columns: "
            + ", ".join(missing)
        )

    train = data.loc[
        data["split"] == "train"
    ].copy()

    validation = data.loc[
        data["split"] == "validation"
    ].copy()

    test = data.loc[
        data["split"] == "test"
    ].copy()

    calibration_mask = (
        train["learner_group_id"]
        .map(calibration_bucket)
        < 20
    )

    fit_data = train.loc[
        ~calibration_mask
    ].copy()

    calibration_data = train.loc[
        calibration_mask
    ].copy()

    fit_learners = set(
        fit_data["learner_group_id"]
    )

    calibration_learners = set(
        calibration_data[
            "learner_group_id"
        ]
    )

    validation_learners = set(
        validation[
            "learner_group_id"
        ]
    )

    test_learners = set(
        test["learner_group_id"]
    )

    if (
        fit_learners
        & calibration_learners
        or fit_learners
        & validation_learners
        or fit_learners
        & test_learners
        or calibration_learners
        & validation_learners
        or calibration_learners
        & test_learners
        or validation_learners
        & test_learners
    ):
        raise RuntimeError(
            "Learner leakage was found "
            "between dataset partitions."
        )

    candidates = {
        "logistic_c_0_25": (
            LogisticRegression(
                C=0.25,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "logistic_c_1": (
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "logistic_c_4": (
            LogisticRegression(
                C=4.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest_depth_8": (
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight=(
                    "balanced_subsample"
                ),
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest_depth_12": (
            RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight=(
                    "balanced_subsample"
                ),
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }

    candidate_metrics = {}
    fitted = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(),
                ),
                (
                    "classifier",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            fit_data[FEATURES],
            fit_data[TARGET],
        )

        probabilities = (
            pipeline.predict_proba(
                validation[FEATURES]
            )[:, 1]
        )

        candidate_metrics[name] = metrics(
            validation[TARGET],
            probabilities,
        )

        fitted[name] = pipeline

    best_logistic = max(
        [
            name
            for name in candidate_metrics
            if name.startswith("logistic_")
        ],
        key=lambda name: (
            candidate_metrics[name][
                "roc_auc"
            ],
            -candidate_metrics[name][
                "brier_score"
            ],
        ),
    )

    best_forest = max(
        [
            name
            for name in candidate_metrics
            if name.startswith(
                "random_forest_"
            )
        ],
        key=lambda name: (
            candidate_metrics[name][
                "roc_auc"
            ],
            -candidate_metrics[name][
                "brier_score"
            ],
        ),
    )

    selected = {
        "logistic_regression": (
            fitted[best_logistic]
        ),
        "random_forest": (
            fitted[best_forest]
        ),
    }

    calibrated_models = {}
    validation_results = {}
    test_results = {}

    for family, base_model in (
        selected.items()
    ):
        calibrated = (
            CalibratedClassifierCV(
                FrozenEstimator(
                    base_model
                ),
                method="sigmoid",
            )
        )

        calibrated.fit(
            calibration_data[FEATURES],
            calibration_data[TARGET],
        )

        validation_probability = (
            calibrated.predict_proba(
                validation[FEATURES]
            )[:, 1]
        )

        test_probability = (
            calibrated.predict_proba(
                test[FEATURES]
            )[:, 1]
        )

        calibrated_models[
            family
        ] = calibrated

        validation_results[
            family
        ] = metrics(
            validation[TARGET],
            validation_probability,
        )

        test_results[
            family
        ] = metrics(
            test[TARGET],
            test_probability,
        )

        joblib.dump(
            calibrated,
            output_dir
            / f"{family}.joblib",
            compress=3,
        )

    champion = max(
        validation_results,
        key=lambda name: (
            validation_results[name][
                "roc_auc"
            ],
            validation_results[name][
                "average_precision"
            ],
            -validation_results[name][
                "brier_score"
            ],
        ),
    )

    joblib.dump(
        calibrated_models[champion],
        output_dir
        / "recall_model_champion.joblib",
        compress=3,
    )

    report = {
        "synthetic_data": (
            "synthetic"
            in dataset_path.name.lower()
        ),
        "scikit_learn_version": (
            sklearn.__version__
        ),
        "selected_candidates": {
            "logistic_regression": (
                best_logistic
            ),
            "random_forest": (
                best_forest
            ),
        },
        "validation_metrics": (
            validation_results
        ),
        "test_metrics": (
            test_results
        ),
        "champion_selected_using_validation": (
            champion
        ),
    }

    (
        output_dir
        / "training_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "model_name": (
            "slsl_receptive_future_recall"
        ),
        "model_version": "1.0.0",
        "champion_family": champion,
        "target": TARGET,
        "positive_class": 1,
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "all_model_features": FEATURES,
        "default_decision_threshold": (
            0.5
        ),
        "training_data_is_synthetic": (
            "synthetic"
            in dataset_path.name.lower()
        ),
    }

    (
        output_dir
        / "model_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
