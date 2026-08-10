from datetime import datetime, timezone

import pandas as pd

from app.ml.dataset_schema import FEATURE_COLUMNS, TARGET_COLUMN


def build_quality_report(
    *, training: pd.DataFrame, audit: pd.DataFrame, unlabeled: pd.DataFrame,
    source_event_count: int, eligible_event_count: int, schema_version: str,
    min_delay_hours: int, max_delay_days: int, minimum_training_rows: int,
    minimum_class_rows: int,
) -> dict:
    row_count = len(training)
    counts = (training[TARGET_COLUMN].value_counts(dropna=False).to_dict()
              if TARGET_COLUMN in training.columns else {})
    positives, negatives = int(counts.get(1, 0)), int(counts.get(0, 0))
    split_counts = (training["split"].value_counts().to_dict()
                    if not training.empty and "split" in training else {})
    learners = {name: set(training.loc[training["split"] == name,
                                      "learner_group_id"].astype(str))
                if not training.empty else set()
                for name in ("train", "validation", "test")}
    overlap = bool((learners["train"] & learners["validation"])
                   or (learners["train"] & learners["test"])
                   or (learners["validation"] & learners["test"]))
    if audit.empty:
        targets_after = True
    else:
        anchors = pd.to_datetime(audit["anchor_completed_at"], utc=True, errors="coerce")
        targets = pd.to_datetime(audit["target_completed_at"], utc=True, errors="coerce")
        valid = anchors.notna() & targets.notna()
        targets_after = bool((targets[valid] > anchors[valid]).all())
    missing = {column: (int(training[column].isna().sum())
                        if column in training else row_count)
               for column in FEATURE_COLUMNS}
    duplicates = (int(training["row_id"].duplicated().sum())
                  if not training.empty and "row_id" in training else 0)
    target_in_features = TARGET_COLUMN in FEATURE_COLUMNS
    ready = all([
        row_count >= minimum_training_rows,
        positives >= minimum_class_rows,
        negatives >= minimum_class_rows,
        all(split_counts.get(name, 0) > 0 for name in ("train", "validation", "test")),
        not overlap, not target_in_features, targets_after, duplicates == 0,
    ])
    warnings = []
    if row_count == 0:
        warnings.append("No delayed-recall labels are available yet.")
    if row_count < minimum_training_rows:
        warnings.append("The labeled dataset is smaller than the configured training readiness threshold.")
    if positives < minimum_class_rows:
        warnings.append("The positive recall class has insufficient examples.")
    if negatives < minimum_class_rows:
        warnings.append("The recall-failure class has insufficient examples.")
    for name in ("train", "validation", "test"):
        if split_counts.get(name, 0) == 0:
            warnings.append(f"The {name} split is empty. This is expected when there are very few learners.")
    if overlap:
        warnings.append("At least one learner appears in multiple dataset splits.")
    if not targets_after:
        warnings.append("At least one target attempt does not occur after its anchor.")
    if duplicates:
        warnings.append("Duplicate dataset row IDs were found.")
    total = max(positives + negatives, 1)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": schema_version,
        "label_definition": {
            "target": TARGET_COLUMN, "minimum_delay_hours": min_delay_hours,
            "maximum_delay_days": max_delay_days,
            "positive_definition": "First eligible future receptive attempt was correct.",
            "negative_definition": "First eligible future receptive attempt was incorrect or skipped.",
        },
        "source": {
            "source_events": source_event_count,
            "eligible_receptive_events": eligible_event_count,
            "labeled_rows": row_count, "unlabeled_rows": len(unlabeled),
            "unique_learners": int(training["learner_group_id"].nunique()) if not training.empty else 0,
            "unique_signs": int(training["sign_group_id"].nunique()) if not training.empty else 0,
        },
        "class_balance": {
            "positive_rows": positives, "negative_rows": negatives,
            "positive_rate": round(positives / total, 6) if row_count else 0.0,
            "negative_rate": round(negatives / total, 6) if row_count else 0.0,
        },
        "splits": {
            "row_counts": {name: int(split_counts.get(name, 0)) for name in learners},
            "learner_counts": {name: len(values) for name, values in learners.items()},
            "cross_split_learner_overlap": overlap,
        },
        "quality_checks": {
            "duplicate_row_ids": duplicates,
            "target_column_in_features": target_in_features,
            "all_targets_after_anchors": targets_after,
            "raw_database_identifiers_exported": False,
            "feature_missing_values": missing,
        },
        "readiness": {
            "minimum_training_rows": minimum_training_rows,
            "minimum_rows_per_class": minimum_class_rows,
            "ready_for_model_training": ready,
            "warnings": warnings,
        },
    }
