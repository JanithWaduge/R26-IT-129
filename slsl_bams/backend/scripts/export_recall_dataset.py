import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pymongo import AsyncMongoClient

from app.core.config import get_settings
from app.ml.dataset_schema import (
    AUDIT_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS,
    METADATA_COLUMNS, NUMERIC_FEATURE_COLUMNS, TARGET_COLUMN,
    TRAINING_COLUMNS, UNLABELED_COLUMNS,
)
from app.ml.mongo_source import load_receptive_recall_events
from app.ml.pseudonym import Pseudonymizer
from app.ml.quality_report import build_quality_report
from app.ml.recall_dataset import RecallDatasetBuilder


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the receptive delayed-recall machine-learning dataset."
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-delay-hours", type=int, default=None)
    parser.add_argument("--max-delay-days", type=int, default=None)
    return parser.parse_args()


async def export_dataset(args: argparse.Namespace) -> None:
    settings = get_settings()
    if settings.ml_pseudonym_secret is None:
        raise RuntimeError(
            "ML_PSEUDONYM_SECRET must be set before exporting research data."
        )
    min_hours = (args.min_delay_hours if args.min_delay_hours is not None
                 else settings.ml_min_recall_delay_hours)
    max_days = (args.max_delay_days if args.max_delay_days is not None
                else settings.ml_max_recall_delay_days)
    if max_days * 24 <= min_hours:
        raise ValueError("Maximum recall delay must be greater than minimum recall delay.")
    pseudonymizer = Pseudonymizer(
        settings.ml_pseudonym_secret.get_secret_value()
    )
    client = AsyncMongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        await client.admin.command("ping")
        events = await load_receptive_recall_events(
            client[settings.mongo_database]
        )
        result = RecallDatasetBuilder(
            pseudonymizer=pseudonymizer,
            min_delay=timedelta(hours=min_hours),
            max_delay=timedelta(days=max_days),
        ).build(events, now=datetime.now(timezone.utc))
    finally:
        await client.close()

    root = Path(args.output_dir or settings.ml_export_directory)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / f"receptive_recall_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    training = pd.DataFrame(result.labeled_rows).reindex(columns=TRAINING_COLUMNS)
    audit = pd.DataFrame(result.audit_rows).reindex(columns=AUDIT_COLUMNS)
    unlabeled = pd.DataFrame(result.unlabeled_rows).reindex(columns=UNLABELED_COLUMNS)
    paths = {
        "training": output / "receptive_recall_training.csv",
        "audit": output / "receptive_recall_audit.csv",
        "unlabeled": output / "receptive_recall_unlabeled.csv",
        "quality_report": output / "dataset_report.json",
    }
    training.to_csv(paths["training"], index=False, encoding="utf-8")
    audit.to_csv(paths["audit"], index=False, encoding="utf-8")
    unlabeled.to_csv(paths["unlabeled"], index=False, encoding="utf-8")
    report = build_quality_report(
        training=training, audit=audit, unlabeled=unlabeled,
        source_event_count=result.source_event_count,
        eligible_event_count=result.eligible_event_count,
        schema_version=settings.ml_dataset_schema_version,
        min_delay_hours=min_hours, max_delay_days=max_days,
        minimum_training_rows=settings.ml_min_training_rows,
        minimum_class_rows=settings.ml_min_class_rows,
    )
    manifest = {
        "schema_version": settings.ml_dataset_schema_version,
        "dataset_type": "receptive delayed-recall binary classification",
        "target_column": TARGET_COLUMN,
        "metadata_columns": list(METADATA_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURE_COLUMNS),
        "model_feature_columns": list(FEATURE_COLUMNS),
        "do_not_use_as_model_features": [
            "row_id", "learner_group_id", "sign_group_id", "split",
            "anchor_event_id", "anchor_session_id", "anchor_question_id",
            "anchor_completed_at", "target_event_id", "target_session_id",
            "target_question_id", "target_completed_at", "label_delay_hours",
            "target_quality_score",
        ],
        "split_policy": {
            "unit": "learner", "method": "deterministic keyed-hash group split",
            "train_percent": 70, "validation_percent": 15, "test_percent": 15,
        },
        "privacy": {
            "raw_student_ids_exported": False,
            "raw_sign_ids_exported": False,
            "raw_session_ids_exported": False,
            "pseudonym_method": "HMAC-SHA256",
        },
        "files": {name: path.name for name, path in paths.items()},
    }
    paths["quality_report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.txt").write_text(str(output.resolve()), encoding="utf-8")
    print("\n" + "=" * 72)
    print("SLSL-BAMS RECEPTIVE RECALL DATASET EXPORT")
    print("=" * 72)
    print(f"Source events       : {result.source_event_count}")
    print(f"Eligible events     : {result.eligible_event_count}")
    print(f"Labeled rows        : {len(training)}")
    print(f"Unlabeled rows      : {len(unlabeled)}")
    print(f"Ready for training  : {report['readiness']['ready_for_model_training']}")
    print(f"Output directory    : {output.resolve()}")
    print("=" * 72)
    if report["readiness"]["warnings"]:
        print("\nWarnings:")
        for warning in report["readiness"]["warnings"]:
            print(f"  - {warning}")


def main() -> None:
    asyncio.run(export_dataset(parse_arguments()))


if __name__ == "__main__":
    main()
