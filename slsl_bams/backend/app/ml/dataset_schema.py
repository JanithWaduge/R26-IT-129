DATASET_SCHEMA_VERSION = "slsl-receptive-recall-v1"

METADATA_COLUMNS = ("row_id", "learner_group_id", "sign_group_id", "split")
CATEGORICAL_FEATURE_COLUMNS = (
    "category_code", "prompt_language", "selection_strategy",
    "selection_reason", "anchor_outcome",
)
NUMERIC_FEATURE_COLUMNS = (
    "anchor_correct", "anchor_was_skipped", "anchor_quality_score",
    "anchor_quality_band", "response_time_ms", "retry_count", "hint_count",
    "sign_difficulty", "was_due", "days_overdue", "selection_priority",
    "pre_mastery_score", "post_mastery_score", "pre_total_reviews",
    "pre_successful_reviews", "pre_failure_count", "pre_average_quality_score",
    "pre_average_response_time_ms", "hours_since_previous_review",
    "pre_ease_factor", "post_ease_factor", "pre_interval_days",
    "post_interval_days", "pre_repetition_count", "post_repetition_count",
    "pre_lapse_count", "post_lapse_count", "scheduled_delay_hours",
    "learner_prior_attempts", "learner_prior_accuracy",
    "learner_prior_average_quality",
)
FEATURE_COLUMNS = CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS
TARGET_COLUMN = "future_recall_success"
TRAINING_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,)
AUDIT_COLUMNS = (
    "row_id", "learner_group_id", "sign_group_id", "split",
    "anchor_event_id", "anchor_session_id", "anchor_question_id",
    "anchor_completed_at", "target_event_id", "target_session_id",
    "target_question_id", "target_completed_at", "label_delay_hours",
    "target_quality_score", TARGET_COLUMN,
)
UNLABELED_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + (
    "anchor_event_id", "anchor_completed_at", "unlabeled_reason",
)
