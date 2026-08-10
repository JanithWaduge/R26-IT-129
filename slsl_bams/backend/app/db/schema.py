LOCALIZED_TEXT_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "english",
        "sinhala",
        "tamil",
    ],
    "additionalProperties": False,
    "properties": {
        "english": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "sinhala": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "tamil": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 500,
        },
    },
}


CURRICULUM_CATEGORY_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Curriculum category document",
        "required": [
            "code",
            "name",
            "description",
            "display_order",
            "icon_key",
            "validation_status",
            "is_active",
            "created_at",
            "updated_at",
            "adaptive_priority_weight",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "code": {
                "bsonType": "string",
                "minLength": 2,
                "maxLength": 50,
            },
            "name": LOCALIZED_TEXT_SCHEMA,
            "description": LOCALIZED_TEXT_SCHEMA,
            "display_order": {
                "bsonType": "int",
                "minimum": 0,
            },
            "icon_key": {
                "bsonType": "string",
                "minLength": 1,
                "maxLength": 50,
            },
            "validation_status": {
                "enum": [
                    "provisional",
                    "teacher_approved",
                ],
            },
            "is_active": {
                "bsonType": "bool",
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
            "adaptive_priority_weight": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0.50,
                "maximum": 2.00,
            },
        },
    }
}


CURRICULUM_COMPETENCY_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Curriculum competency document",
        "required": [
            "category_id",
            "code",
            "title",
            "description",
            "grade_levels",
            "display_order",
            "receptive_mastery_threshold",
            "productive_mastery_threshold",
            "validation_status",
            "is_active",
            "created_at",
            "updated_at",
            "adaptive_priority_weight",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "category_id": {
                "bsonType": "objectId",
            },
            "code": {
                "bsonType": "string",
                "minLength": 2,
                "maxLength": 80,
            },
            "title": LOCALIZED_TEXT_SCHEMA,
            "description": LOCALIZED_TEXT_SCHEMA,
            "grade_levels": {
                "bsonType": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "bsonType": "string",
                    "minLength": 1,
                    "maxLength": 50,
                },
            },
            "display_order": {
                "bsonType": "int",
                "minimum": 0,
            },
            "receptive_mastery_threshold": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
                "maximum": 1,
            },
            "productive_mastery_threshold": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
                "maximum": 1,
            },
            "validation_status": {
                "enum": [
                    "provisional",
                    "teacher_approved",
                ],
            },
            "is_active": {
                "bsonType": "bool",
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
            "adaptive_priority_weight": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0.50,
                "maximum": 2.00,
            },
        },
    }
}


SIGN_MEDIA_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "source_type",
        "uri",
        "thumbnail_uri",
        "duration_ms",
        "objective_source",
    ],
    "additionalProperties": False,
    "properties": {
        "source_type": {
            "enum": [
                "pending",
                "video",
                "avatar_animation",
            ],
        },
        "uri": {
            "bsonType": [
                "string",
                "null",
            ],
        },
        "thumbnail_uri": {
            "bsonType": [
                "string",
                "null",
            ],
        },
        "duration_ms": {
            "bsonType": [
                "int",
                "null",
            ],
            "minimum": 0,
        },
        "objective_source": {
            "enum": [
                "pending",
                "objective_1",
                "teacher_upload",
            ],
        },
    },
}


SIGN_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Sri Lankan Sign Language concept document",
        "required": [
            "code",
            "gloss",
            "meanings",
            "category_id",
            "competency_ids",
            "difficulty",
            "tags",
            "search_terms",
            "prerequisite_sign_ids",
            "media",
            "content_status",
            "validation_status",
            "is_active",
            "created_at",
            "updated_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "code": {
                "bsonType": "string",
                "minLength": 2,
                "maxLength": 80,
            },
            "gloss": {
                "bsonType": "string",
                "minLength": 1,
                "maxLength": 100,
            },
            "meanings": LOCALIZED_TEXT_SCHEMA,
            "category_id": {
                "bsonType": "objectId",
            },
            "competency_ids": {
                "bsonType": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "bsonType": "objectId",
                },
            },
            "difficulty": {
                "bsonType": "int",
                "minimum": 1,
                "maximum": 5,
            },
            "tags": {
                "bsonType": "array",
                "uniqueItems": True,
                "items": {
                    "bsonType": "string",
                    "minLength": 1,
                    "maxLength": 50,
                },
            },
            "search_terms": {
                "bsonType": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "bsonType": "string",
                    "minLength": 1,
                    "maxLength": 120,
                },
            },
            "prerequisite_sign_ids": {
                "bsonType": "array",
                "uniqueItems": True,
                "items": {
                    "bsonType": "objectId",
                },
            },
            "media": SIGN_MEDIA_SCHEMA,
            "content_status": {
                "enum": [
                    "development",
                    "approved",
                    "archived",
                ],
            },
            "validation_status": {
                "enum": [
                    "provisional",
                    "teacher_approved",
                ],
            },
            "is_active": {
                "bsonType": "bool",
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
        },
    }
}

STUDENT_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Student document validation",
        "required": [
            "full_name",
            "email",
            "preferred_language",
            "grade_level",
            "is_active",
            "created_at",
            "updated_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
                "description": "MongoDB-generated student identifier.",
            },
            "full_name": {
                "bsonType": "string",
                "minLength": 2,
                "maxLength": 120,
            },
            "email": {
                "bsonType": "string",
                "minLength": 5,
                "maxLength": 254,
            },
            "preferred_language": {
                "enum": [
                    "sinhala",
                    "tamil",
                    "english",
                ],
            },
            "grade_level": {
                "bsonType": "string",
                "minLength": 1,
                "maxLength": 50,
            },
            "is_active": {
                "bsonType": "bool",
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
        },
    }
}


MIGRATION_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Database migration record",
        "required": [
            "version",
            "name",
            "applied_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "version": {
                "bsonType": "int",
                "minimum": 1,
            },
            "name": {
                "bsonType": "string",
                "minLength": 1,
            },
            "applied_at": {
                "bsonType": "date",
            },
        },
    }
}

USER_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Authentication user document",
        "required": [
            "student_id",
            "email",
            "password_hash",
            "role",
            "is_active",
            "token_version",
            "created_at",
            "updated_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "email": {
                "bsonType": "string",
                "minLength": 5,
                "maxLength": 254,
            },
            "password_hash": {
                "bsonType": "string",
                "minLength": 20,
            },
            "role": {
                "enum": [
                    "student",
                    "teacher",
                    "admin",
                ],
            },
            "is_active": {
                "bsonType": "bool",
            },
            "token_version": {
                "bsonType": "int",
                "minimum": 0,
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
        },
    }
}


REFRESH_SESSION_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Refresh-token session document",
        "required": [
            "user_id",
            "jti_hash",
            "family_id",
            "expires_at",
            "revoked_at",
            "replaced_by_jti_hash",
            "created_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "user_id": {
                "bsonType": "objectId",
            },
            "jti_hash": {
                "bsonType": "string",
                "minLength": 64,
                "maxLength": 64,
            },
            "family_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "expires_at": {
                "bsonType": "date",
            },
            "revoked_at": {
                "bsonType": [
                    "date",
                    "null",
                ],
            },
            "replaced_by_jti_hash": {
                "bsonType": [
                    "string",
                    "null",
                ],
            },
            "created_at": {
                "bsonType": "date",
            },
        },
    }
}


QUIZ_LOCALIZED_TEXT_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "english",
        "sinhala",
        "tamil",
    ],
    "additionalProperties": False,
    "properties": {
        "english": {
            "bsonType": "string",
        },
        "sinhala": {
            "bsonType": "string",
        },
        "tamil": {
            "bsonType": "string",
        },
    },
}


QUIZ_OPTION_SNAPSHOT_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "sign_id",
        "meanings",
    ],
    "additionalProperties": False,
    "properties": {
        "sign_id": {
            "bsonType": "objectId",
        },
        "meanings": QUIZ_LOCALIZED_TEXT_SCHEMA,
    },
}


QUIZ_TOP_PREDICTION_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "sign_id",
        "sign_code",
        "confidence",
    ],
    "additionalProperties": False,
    "properties": {
        "sign_id": {
            "bsonType": [
                "objectId",
                "null",
            ],
        },
        "sign_code": {
            "bsonType": [
                "string",
                "null",
            ],
        },
        "confidence": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 1,
        },
    },
}


QUIZ_RECOGNITION_RESULT_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "request_id",
        "model_name",
        "model_version",
        "predicted_sign_id",
        "predicted_code",
        "confidence",
        "inference_time_ms",
        "top_predictions",
    ],
    "additionalProperties": False,
    "properties": {
        "request_id": {
            "bsonType": "string",
            "minLength": 36,
            "maxLength": 36,
        },
        "model_name": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 100,
        },
        "model_version": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 50,
        },
        "predicted_sign_id": {
            "bsonType": [
                "objectId",
                "null",
            ],
        },
        "predicted_code": {
            "bsonType": [
                "string",
                "null",
            ],
        },
        "confidence": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 1,
        },
        "inference_time_ms": {
            "bsonType": "int",
            "minimum": 0,
        },
        "top_predictions": {
            "bsonType": "array",
            "items": QUIZ_TOP_PREDICTION_SCHEMA,
        },
    },
}


QUIZ_INTERACTION_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "event_id",
        "event_type",
        "attempt_number",
        "selected_sign_id",
        "recognition_result",
        "is_correct",
        "client_response_time_ms",
        "server_response_time_ms",
        "created_at",
    ],
    "additionalProperties": False,
    "properties": {
        "event_id": {
            "bsonType": "string",
            "minLength": 36,
            "maxLength": 36,
        },
        "event_type": {
            "enum": [
                "answer",
                "hint",
                "skip",
            ],
        },
        "attempt_number": {
            "bsonType": "int",
            "minimum": 0,
        },
        "selected_sign_id": {
            "bsonType": [
                "objectId",
                "null",
            ],
        },
        "recognition_result": {
            "bsonType": [
                "object",
                "null",
            ],
        },
        "is_correct": {
            "bsonType": [
                "bool",
                "null",
            ],
        },
        "client_response_time_ms": {
            "bsonType": [
                "int",
                "null",
            ],
            "minimum": 0,
        },
        "server_response_time_ms": {
            "bsonType": [
                "int",
                "null",
            ],
            "minimum": 0,
        },
        "created_at": {
            "bsonType": "date",
        },
    },
}


QUIZ_PROMPT_SNAPSHOT_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "meanings",
        "category_name",
        "difficulty",
        "media",
    ],
    "additionalProperties": False,
    "properties": {
        "meanings": QUIZ_LOCALIZED_TEXT_SCHEMA,
        "category_name": QUIZ_LOCALIZED_TEXT_SCHEMA,
        "difficulty": {
            "bsonType": "int",
            "minimum": 1,
            "maximum": 5,
        },
        "media": SIGN_MEDIA_SCHEMA,
    },
}


QUIZ_QUESTION_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
    "question_id",
    "sign_id",
    "expected_code",
    "direction",
    "order_index",
    "prompt_snapshot",
    "option_snapshots",
    "status",
    "started_at",
    "completed_at",
    "interactions",
    "hint_count",
    "retry_count",
    "final_correct",
    "final_response_time_ms",
    "mastery_event_id",
    "mastery_applied_at",
    "quality_score",
    "selection_priority",
    "selection_reason",
    "was_due",
    "days_overdue",
    ],
    "additionalProperties": False,
    "properties": {
        "question_id": {
            "bsonType": "string",
            "minLength": 36,
            "maxLength": 36,
        },
        "sign_id": {
            "bsonType": "objectId",
        },
        "expected_code": {
            "bsonType": "string",
            "minLength": 1,
        },
        "direction": {
            "enum": [
                "receptive",
                "productive",
            ],
        },
        "order_index": {
            "bsonType": "int",
            "minimum": 0,
        },
        "prompt_snapshot": QUIZ_PROMPT_SNAPSHOT_SCHEMA,
        "option_snapshots": {
            "bsonType": "array",
            "items": QUIZ_OPTION_SNAPSHOT_SCHEMA,
        },
        "status": {
            "enum": [
                "pending",
                "correct",
                "incorrect",
                "skipped",
            ],
        },
        "started_at": {
            "bsonType": [
                "date",
                "null",
            ],
        },
        "completed_at": {
            "bsonType": [
                "date",
                "null",
            ],
        },
        "interactions": {
            "bsonType": "array",
            "items": QUIZ_INTERACTION_SCHEMA,
        },
        "hint_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "retry_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "final_correct": {
            "bsonType": [
                "bool",
                "null",
            ],
        },
        "final_response_time_ms": {
            "bsonType": [
                "int",
                "null",
            ],
            "minimum": 0,
        },
         "mastery_event_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "mastery_applied_at": {
                "bsonType": [
                    "date",
                    "null",
                ],
            },
            "quality_score": {
                "bsonType": [
                    "double",
                    "int",
                    "null",
                ],
                "minimum": 0,
                "maximum": 5,
            },
            "selection_priority": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
                "maximum": 100,
            },
            "selection_reason": {
                "bsonType": "string",
                "minLength": 1,
                "maxLength": 100,
            },
            "was_due": {
                "bsonType": "bool",
            },
            "days_overdue": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
            },
    },
}


QUIZ_SUMMARY_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "correct_count",
        "incorrect_count",
        "skipped_count",
        "hints_used",
        "total_retries",
        "receptive_correct",
        "productive_correct",
    ],
    "additionalProperties": False,
    "properties": {
        "correct_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "incorrect_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "skipped_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "hints_used": {
            "bsonType": "int",
            "minimum": 0,
        },
        "total_retries": {
            "bsonType": "int",
            "minimum": 0,
        },
        "receptive_correct": {
            "bsonType": "int",
            "minimum": 0,
        },
        "productive_correct": {
            "bsonType": "int",
            "minimum": 0,
        },
    },
}


QUIZ_SESSION_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Quiz session document",
        "required": [
            "student_id",
            "client_request_id",
            "mode",
            "prompt_language",
            "status",
            "filters",
            "question_count",
            "current_question_index",
            "questions",
            "summary",
            "selection_seed",
            "version",
            "started_at",
            "expires_at",
            "completed_at",
            "created_at",
            "updated_at",
            "selection_strategy",
            "experiment_name",
            "experiment_arm",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "client_request_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "mode": {
                "enum": [
                    "receptive",
                    "productive",
                    "mixed",
                ],
            },
            "prompt_language": {
                "enum": [
                    "english",
                    "sinhala",
                    "tamil",
                ],
            },
            "status": {
                "enum": [
                    "in_progress",
                    "completed",
                    "abandoned",
                    "expired",
                ],
            },
            "filters": {
                "bsonType": "object",
                "required": [
                    "category_id",
                    "competency_id",
                    "difficulty",
                ],
                "additionalProperties": False,
                "properties": {
                    "category_id": {
                        "bsonType": [
                            "objectId",
                            "null",
                        ],
                    },
                    "competency_id": {
                        "bsonType": [
                            "objectId",
                            "null",
                        ],
                    },
                    "difficulty": {
                        "bsonType": [
                            "int",
                            "null",
                        ],
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
            },
            "question_count": {
                "bsonType": "int",
                "minimum": 1,
                "maximum": 20,
            },
            "current_question_index": {
                "bsonType": "int",
                "minimum": 0,
            },
            "questions": {
                "bsonType": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": QUIZ_QUESTION_SCHEMA,
            },
            "summary": QUIZ_SUMMARY_SCHEMA,
            "selection_seed": {
                "bsonType": "long",
            },
            "version": {
                "bsonType": "int",
                "minimum": 1,
            },
            "started_at": {
                "bsonType": "date",
            },
            "expires_at": {
                "bsonType": "date",
            },
            "completed_at": {
                "bsonType": [
                    "date",
                    "null",
                ],
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
           "selection_strategy": {
                "enum": [
                "adaptive",
                "due_only",
                "random_control",
                "legacy_random",
    ],
},
            "experiment_name": {"bsonType": ["string", "null"]},
            "experiment_arm": {"enum": [
                "random_control", "modified_sm2", "hybrid_ml",
                "not_enrolled", "legacy_unassigned",
            ]},
        },
    }
}

MASTERY_STATUS_VALUES = [
    "new",
    "very_weak",
    "weak",
    "learning",
    "proficient",
    "mastered",
]


MASTERY_BALANCE_VALUES = [
    "unassessed",
    "receptive_unassessed",
    "productive_unassessed",
    "receptive_weaker",
    "productive_weaker",
    "balanced",
]


MASTERY_DIRECTION_STATE_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "score",
        "status",
        "total_reviews",
        "successful_reviews",
        "independent_successes",
        "assisted_successes",
        "failure_count",
        "average_quality_score",
        "last_quality_score",
        "last_quality_band",
        "response_samples",
        "average_response_time_ms",
        "confidence_samples",
        "average_recognition_confidence",
        "last_recognition_confidence",
        "last_reviewed_at",
        "last_session_id",
        "last_question_id",
        "ease_factor",
        "previous_interval_days",
        "interval_days",
        "repetition_count",
        "lapse_count",
        "next_review_at",
        "last_scheduled_at",
        "last_schedule_reason",
        "base_sm2_interval_days",
        "last_ml_probability",
        "last_ml_adjustment_factor",
        "last_ml_model_version",
        "last_ml_status",
        "last_ml_event_id",
        "last_ml_applied_at",
    ],
    "additionalProperties": False,
    "properties": {
        "score": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 1,
        },
        "status": {
            "enum": MASTERY_STATUS_VALUES,
        },
        "total_reviews": {
            "bsonType": "int",
            "minimum": 0,
        },
        "successful_reviews": {
            "bsonType": "int",
            "minimum": 0,
        },
        "independent_successes": {
            "bsonType": "int",
            "minimum": 0,
        },
        "assisted_successes": {
            "bsonType": "int",
            "minimum": 0,
        },
        "failure_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "average_quality_score": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 5,
        },
        "last_quality_score": {
            "bsonType": [
                "double",
                "int",
                "null",
            ],
            "minimum": 0,
            "maximum": 5,
        },
        "last_quality_band": {
            "bsonType": [
                "int",
                "null",
            ],
            "minimum": 0,
            "maximum": 5,
        },
        "response_samples": {
            "bsonType": "int",
            "minimum": 0,
        },
        "average_response_time_ms": {
            "bsonType": [
                "double",
                "int",
                "null",
            ],
            "minimum": 0,
        },
        "confidence_samples": {
            "bsonType": "int",
            "minimum": 0,
        },
        "average_recognition_confidence": {
            "bsonType": [
                "double",
                "int",
                "null",
            ],
            "minimum": 0,
            "maximum": 1,
        },
        "last_recognition_confidence": {
            "bsonType": [
                "double",
                "int",
                "null",
            ],
            "minimum": 0,
            "maximum": 1,
        },
        "last_reviewed_at": {
            "bsonType": [
                "date",
                "null",
            ],
        },
        "last_session_id": {
            "bsonType": [
                "objectId",
                "null",
            ],
        },
        "last_question_id": {
            "bsonType": [
                "string",
                "null",
            ],
        },
        "ease_factor": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 1.30,
            "maximum": 3.50,
        },
        "previous_interval_days": {
            "bsonType": "int",
            "minimum": 0,
        },
        "interval_days": {
            "bsonType": "int",
            "minimum": 0,
        },
        "base_sm2_interval_days": {"bsonType": "int", "minimum": 0},
        "last_ml_probability": {"bsonType": ["double", "int", "null"]},
        "last_ml_adjustment_factor": {"bsonType": ["double", "int", "null"]},
        "last_ml_model_version": {"bsonType": ["string", "null"]},
        "last_ml_status": {"enum": [
            "not_evaluated", "applied", "disabled", "unavailable",
            "invalid_features", "prediction_failed", "stale_event",
        ]},
        "last_ml_event_id": {"bsonType": ["string", "null"]},
        "last_ml_applied_at": {"bsonType": ["date", "null"]},
        "repetition_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "lapse_count": {
            "bsonType": "int",
            "minimum": 0,
        },
        "next_review_at": {
            "bsonType": [
                "date",
                "null",
            ],
        },
        "last_scheduled_at": {
            "bsonType": [
                "date",
                "null",
            ],
        },
        "last_schedule_reason": {
            "enum": [
                "unassessed",
                "first_success",
                "second_success",
                "successful_review",
                "assisted_success",
                "strong_success",
                "lapse",
            ],
        },
    },
}


PROCESSED_MASTERY_EVENT_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "event_id",
        "direction",
        "quality_score",
        "quality_band",
        "before_score",
        "after_score",
        "applied_at",
    ],
    "additionalProperties": False,
    "properties": {
        "event_id": {
            "bsonType": "string",
            "minLength": 36,
            "maxLength": 36,
        },
        "direction": {
            "enum": [
                "receptive",
                "productive",
            ],
        },
        "quality_score": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 5,
        },
        "quality_band": {
            "bsonType": "int",
            "minimum": 0,
            "maximum": 5,
        },
        "before_score": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 1,
        },
        "after_score": {
            "bsonType": [
                "double",
                "int",
            ],
            "minimum": 0,
            "maximum": 1,
        },
        "applied_at": {
            "bsonType": "date",
        },
    },
}


MASTERY_RECORD_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Student sign mastery record",
        "required": [
            "student_id",
            "sign_id",
            "receptive",
            "productive",
            "combined_score",
            "overall_status",
            "balance_status",
            "processed_events",
            "version",
            "created_at",
            "updated_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "sign_id": {
                "bsonType": "objectId",
            },
            "receptive": MASTERY_DIRECTION_STATE_SCHEMA,
            "productive": MASTERY_DIRECTION_STATE_SCHEMA,
            "combined_score": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
                "maximum": 1,
            },
            "overall_status": {
                "enum": MASTERY_STATUS_VALUES,
            },
            "balance_status": {
                "enum": MASTERY_BALANCE_VALUES,
            },
            "processed_events": {
                "bsonType": "array",
                "items": (
                    PROCESSED_MASTERY_EVENT_SCHEMA
                ),
            },
            "version": {
                "bsonType": "int",
                "minimum": 1,
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
        },
    }
}


MASTERY_QUALITY_COMPONENTS_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "base_score",
        "retry_penalty",
        "hint_penalty",
        "response_time_penalty",
        "confidence_penalty",
        "response_target_ms",
    ],
    "additionalProperties": False,
    "properties": {
        "base_score": {
            "bsonType": [
                "double",
                "int",
            ],
        },
        "retry_penalty": {
            "bsonType": [
                "double",
                "int",
            ],
        },
        "hint_penalty": {
            "bsonType": [
                "double",
                "int",
            ],
        },
        "response_time_penalty": {
            "bsonType": [
                "double",
                "int",
            ],
        },
        "confidence_penalty": {
            "bsonType": [
                "double",
                "int",
            ],
        },
        "response_target_ms": {
            "bsonType": "int",
            "minimum": 0,
        },
    },
}


MASTERY_EVENT_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Mastery update research event",
        "required": [
            "event_id",
            "status",
            "student_id",
            "sign_id",
            "session_id",
            "question_id",
            "direction",
            "question_status",
            "final_correct",
            "retry_count",
            "hint_count",
            "response_time_ms",
            "recognition_confidence",
            "difficulty",
            "quality_score",
            "quality_band",
            "quality_components",
            "algorithm_version",
            "before_state",
            "after_state",
            "created_at",
            "applied_at",
            "scheduler_algorithm_version",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "event_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "status": {
                "enum": [
                    "pending",
                    "applied",
                ],
            },
            "scheduler_algorithm_version": {
                "bsonType": "string",
                "minLength": 1,
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "sign_id": {
                "bsonType": "objectId",
            },
            "session_id": {
                "bsonType": "objectId",
            },
            "question_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "direction": {
                "enum": [
                    "receptive",
                    "productive",
                ],
            },
            "question_status": {
                "enum": [
                    "correct",
                    "incorrect",
                    "skipped",
                ],
            },
            "final_correct": {
                "bsonType": [
                    "bool",
                    "null",
                ],
            },
            "retry_count": {
                "bsonType": "int",
                "minimum": 0,
            },
            "hint_count": {
                "bsonType": "int",
                "minimum": 0,
            },
            "response_time_ms": {
                "bsonType": [
                    "int",
                    "null",
                ],
                "minimum": 0,
            },
            "recognition_confidence": {
                "bsonType": [
                    "double",
                    "int",
                    "null",
                ],
                "minimum": 0,
                "maximum": 1,
            },
            "difficulty": {
                "bsonType": "int",
                "minimum": 1,
                "maximum": 5,
            },
            "quality_score": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
                "maximum": 5,
            },
            "quality_band": {
                "bsonType": "int",
                "minimum": 0,
                "maximum": 5,
            },
            "quality_components": (
                MASTERY_QUALITY_COMPONENTS_SCHEMA
            ),
            "algorithm_version": {
                "bsonType": "string",
                "minLength": 1,
            },
            "before_state": {
                "bsonType": [
                    "object",
                    "null",
                ],
            },
            "after_state": {
                "bsonType": [
                    "object",
                    "null",
                ],
            },
            "created_at": {
                "bsonType": "date",
            },
            "applied_at": {
                "bsonType": [
                    "date",
                    "null",
                ],
            },
        },
    }
}

RECOGNITION_CAPTURE_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "content_type",
        "byte_size",
        "sha256",
        "duration_ms",
        "retained",
        "storage_path",
    ],
    "additionalProperties": False,
    "properties": {
        "content_type": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 100,
        },
        "byte_size": {
            "bsonType": ["int", "long"],
            "minimum": 1,
        },
        "sha256": {
            "bsonType": "string",
            "minLength": 64,
            "maxLength": 64,
        },
        "duration_ms": {
            "bsonType": "int",
            "minimum": 0,
        },
        "retained": {
            "bsonType": "bool",
        },
        "storage_path": {
            "bsonType": [
                "string",
                "null",
            ],
        },
    },
}


RECOGNITION_ATTEMPT_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Productive sign recognition attempt",
        "required": [
            "request_id",
            "student_id",
            "session_id",
            "question_id",
            "expected_sign_id",
            "expected_code",
            "capture",
            "result",
            "accepted",
            "question_finalized",
            "retry_allowed",
            "acceptance_threshold",
            "recognizer_mode",
            "created_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "request_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "session_id": {
                "bsonType": "objectId",
            },
            "question_id": {
                "bsonType": "string",
                "minLength": 36,
                "maxLength": 36,
            },
            "expected_sign_id": {
                "bsonType": "objectId",
            },
            "expected_code": {
                "bsonType": "string",
                "minLength": 1,
            },
            "capture": RECOGNITION_CAPTURE_SCHEMA,
            "result": QUIZ_RECOGNITION_RESULT_SCHEMA,
            "accepted": {
                "bsonType": "bool",
            },
            "question_finalized": {
                "bsonType": "bool",
            },
            "retry_allowed": {
                "bsonType": "bool",
            },
            "acceptance_threshold": {
                "bsonType": [
                    "double",
                    "int",
                ],
                "minimum": 0,
                "maximum": 1,
            },
            "recognizer_mode": {
                "enum": [
                    "mock",
                    "http",
                ],
            },
            "created_at": {
                "bsonType": "date",
            },
        },
    }
}

GAMIFICATION_BADGE_SCHEMA: dict = {
    "bsonType": "object",
    "required": [
        "code",
        "unlocked_at",
        "source_session_id",
    ],
    "additionalProperties": False,
    "properties": {
        "code": {
            "bsonType": "string",
            "minLength": 1,
            "maxLength": 100,
        },
        "unlocked_at": {
            "bsonType": "date",
        },
        "source_session_id": {
            "bsonType": "objectId",
        },
    },
}


GAMIFICATION_PROFILE_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Student gamification profile",
        "required": [
            "student_id",
            "total_xp",
            "level",
            "xp_into_level",
            "xp_to_next_level",
            "current_streak_days",
            "longest_streak_days",
            "last_activity_date",
            "total_quizzes",
            "total_questions",
            "total_correct",
            "total_incorrect",
            "total_skipped",
            "total_hints",
            "total_retries",
            "perfect_quizzes",
            "unlocked_badges",
            "processed_session_ids",
            "version",
            "created_at",
            "updated_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "total_xp": {
                "bsonType": "int",
                "minimum": 0,
            },
            "level": {
                "bsonType": "int",
                "minimum": 1,
            },
            "xp_into_level": {
                "bsonType": "int",
                "minimum": 0,
            },
            "xp_to_next_level": {
                "bsonType": "int",
                "minimum": 1,
            },
            "current_streak_days": {
                "bsonType": "int",
                "minimum": 0,
            },
            "longest_streak_days": {
                "bsonType": "int",
                "minimum": 0,
            },
            "last_activity_date": {
                "bsonType": [
                    "string",
                    "null",
                ],
            },
            "total_quizzes": {
                "bsonType": "int",
                "minimum": 0,
            },
            "total_questions": {
                "bsonType": "int",
                "minimum": 0,
            },
            "total_correct": {
                "bsonType": "int",
                "minimum": 0,
            },
            "total_incorrect": {
                "bsonType": "int",
                "minimum": 0,
            },
            "total_skipped": {
                "bsonType": "int",
                "minimum": 0,
            },
            "total_hints": {
                "bsonType": "int",
                "minimum": 0,
            },
            "total_retries": {
                "bsonType": "int",
                "minimum": 0,
            },
            "perfect_quizzes": {
                "bsonType": "int",
                "minimum": 0,
            },
            "unlocked_badges": {
                "bsonType": "array",
                "items": GAMIFICATION_BADGE_SCHEMA,
            },
            "processed_session_ids": {
                "bsonType": "array",
                "items": {
                    "bsonType": "objectId",
                },
            },
            "version": {
                "bsonType": "int",
                "minimum": 1,
            },
            "created_at": {
                "bsonType": "date",
            },
            "updated_at": {
                "bsonType": "date",
            },
        },
    }
}


GAMIFICATION_EVENT_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Completed quiz reward event",
        "required": [
            "student_id",
            "session_id",
            "status",
            "xp_awarded",
            "badges_awarded",
            "reward_breakdown",
            "total_xp_after",
            "level_after",
            "streak_after",
            "created_at",
            "applied_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {
                "bsonType": "objectId",
            },
            "student_id": {
                "bsonType": "objectId",
            },
            "session_id": {
                "bsonType": "objectId",
            },
            "status": {
                "enum": [
                    "pending",
                    "applied",
                ],
            },
            "xp_awarded": {
                "bsonType": "int",
                "minimum": 0,
            },
            "badges_awarded": {
                "bsonType": "array",
                "items": {
                    "bsonType": "string",
                },
            },
            "reward_breakdown": {
                "bsonType": "object",
                "required": [
                    "question_xp",
                    "due_review_bonus",
                    "completion_bonus",
                    "perfect_bonus",
                ],
                "additionalProperties": False,
                "properties": {
                    "question_xp": {
                        "bsonType": "int",
                        "minimum": 0,
                    },
                    "due_review_bonus": {
                        "bsonType": "int",
                        "minimum": 0,
                    },
                    "completion_bonus": {
                        "bsonType": "int",
                        "minimum": 0,
                    },
                    "perfect_bonus": {
                        "bsonType": "int",
                        "minimum": 0,
                    },
                },
            },
            "total_xp_after": {
                "bsonType": [
                    "int",
                    "null",
                ],
                "minimum": 0,
            },
            "level_after": {
                "bsonType": [
                    "int",
                    "null",
                ],
                "minimum": 1,
            },
            "streak_after": {
                "bsonType": [
                    "int",
                    "null",
                ],
                "minimum": 0,
            },
            "created_at": {
                "bsonType": "date",
            },
            "applied_at": {
                "bsonType": [
                    "date",
                    "null",
                ],
            },
        },
    }
}
ML_RECALL_PREDICTION_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Hybrid receptive recall prediction",
        "required": [
            "event_id", "student_id", "sign_id", "session_id", "question_id",
            "direction", "status", "model_name", "model_version",
            "synthetic_model", "recall_probability", "risk_band",
            "base_interval_days", "adjustment_factor", "final_interval_days",
            "features_hash", "feature_snapshot", "error_message", "created_at",
            "applied_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {"bsonType": "objectId"},
            "event_id": {"bsonType": "string", "minLength": 1},
            "student_id": {"bsonType": "objectId"},
            "sign_id": {"bsonType": "objectId"},
            "session_id": {"bsonType": "objectId"},
            "question_id": {"bsonType": "string", "minLength": 1},
            "direction": {"enum": ["receptive"]},
            "status": {"enum": [
                "pending", "applied", "disabled", "unavailable",
                "invalid_features", "prediction_failed", "stale_event",
            ]},
            "model_name": {"bsonType": ["string", "null"]},
            "model_version": {"bsonType": ["string", "null"]},
            "synthetic_model": {"bsonType": "bool"},
            "recall_probability": {"bsonType": ["double", "int", "null"]},
            "risk_band": {"bsonType": ["string", "null"]},
            "base_interval_days": {"bsonType": "int", "minimum": 1},
            "adjustment_factor": {"bsonType": ["double", "int", "null"]},
            "final_interval_days": {"bsonType": "int", "minimum": 1},
            "features_hash": {"bsonType": "string", "minLength": 64, "maxLength": 64},
            "feature_snapshot": {"bsonType": "object"},
            "error_message": {"bsonType": ["string", "null"]},
            "created_at": {"bsonType": "date"},
            "applied_at": {"bsonType": ["date", "null"]},
        },
    }
}
EXPERIMENT_ASSIGNMENT_VALIDATOR: dict = {
    "$jsonSchema": {
        "bsonType": "object",
        "title": "Scheduler experiment assignment",
        "required": [
            "experiment_name", "student_id", "arm", "stratum",
            "assignment_hash", "is_active", "assigned_at", "created_at", "updated_at",
        ],
        "additionalProperties": False,
        "properties": {
            "_id": {"bsonType": "objectId"},
            "experiment_name": {"bsonType": "string", "minLength": 1},
            "student_id": {"bsonType": "objectId"},
            "arm": {"enum": ["random_control", "modified_sm2", "hybrid_ml"]},
            "stratum": {"bsonType": "string", "minLength": 1},
            "assignment_hash": {"bsonType": "string", "minLength": 64, "maxLength": 64},
            "is_active": {"bsonType": "bool"},
            "assigned_at": {"bsonType": "date"},
            "created_at": {"bsonType": "date"},
            "updated_at": {"bsonType": "date"},
        },
    }
}
