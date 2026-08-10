import json
from pathlib import Path

import pandas as pd
import pytest

from app.analytics.research_dashboard import build_research_dashboard, load_evaluation_bundle


def create_evaluation_directory(root: Path, *, synthetic_model: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    common = {
        "learners": 20, "delayed_labels": 100, "mean_quality_score": 3.8,
        "mean_response_time_ms": 4700, "retained_recalls_per_response_hour": 300.0,
    }
    pd.DataFrame([
        {**common, "arm": "random_control", "retention_rate": .60,
         "retained_recalls_per_100_events": 33.33},
        {**common, "arm": "modified_sm2", "retention_rate": .70,
         "retained_recalls_per_100_events": 41.18},
        {**common, "arm": "hybrid_ml", "retention_rate": .75,
         "retained_recalls_per_100_events": 45.45},
    ]).to_csv(root / "scheduler_arm_summary.csv", index=False)
    pd.DataFrame([{
        "first_arm": "hybrid_ml", "second_arm": "modified_sm2",
        "difference": .05, "ci_95_low": -.01, "ci_95_high": .11,
    }]).to_csv(root / "pairwise_retention_comparisons.csv", index=False)
    labels = [
        {"student_id": f"{arm}-{learner}", "sign_id": f"sign-{learner}",
         "arm": arm, "future_recall_success": int(learner % 4 != 0)}
        for arm in ("random_control", "modified_sm2", "hybrid_ml")
        for learner in range(20)
    ]
    pd.DataFrame(labels).to_csv(root / "delayed_recall_labels.csv", index=False)
    pd.DataFrame([{
        "mean_probability": .75, "observed_retention": .73,
    }]).to_csv(root / "hybrid_calibration.csv", index=False)
    pd.DataFrame([{
        "event_id": "event-1", "synthetic_model": synthetic_model,
        "recall_probability": .75, "future_recall_success": 1,
    }]).to_csv(root / "hybrid_prediction_labels.csv", index=False)
    (root / "evaluation_report.json").write_text(json.dumps({
        "generated_at": "2026-08-02T00:00:00+00:00",
        "experiment_name": "test-scheduler-experiment",
        "label_definition": {"minimum_delay_hours": 24, "maximum_delay_days": 30},
        "hybrid_prediction_metrics": {"rows": 100, "roc_auc": .72},
        "limitations": [],
    }), encoding="utf-8")
    return root


def create_template(path: Path) -> Path:
    path.write_text(
        "<h1>{{ report_title }}</h1><div>{{ experiment_name }}</div>"
        "{% for warning in warnings %}<p>{{ warning }}</p>{% endfor %}"
        "{{ arm_table }}{{ comparison_table }}<img src='{{ charts.retention }}'>",
        encoding="utf-8",
    )
    return path


def build(tmp_path: Path, *, synthetic_model: bool = False):
    source = create_evaluation_directory(tmp_path / "evaluation", synthetic_model=synthetic_model)
    return build_research_dashboard(
        bundle=load_evaluation_bundle(source),
        template_path=create_template(tmp_path / "template.html"),
        output_directory=tmp_path / "report", report_title="Test Report",
        minimum_learners_per_arm=15, include_raw_labels=False,
    )


def test_dashboard_is_generated(tmp_path: Path) -> None:
    result = build(tmp_path)
    for path in (result.dashboard_path, result.summary_path, result.manifest_path,
                 result.checksum_path, result.zip_path):
        assert path.exists()
    html = result.dashboard_path.read_text(encoding="utf-8")
    assert "Test Report" in html
    assert "test-scheduler-experiment" in html


def test_synthetic_model_warning(tmp_path: Path) -> None:
    result = build(tmp_path, synthetic_model=True)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["synthetic_model_used"] is True
    assert summary["ready_for_interpretation"] is False


def test_raw_labels_are_excluded_by_default(tmp_path: Path) -> None:
    result = build(tmp_path)
    assert not (result.output_directory / "data" / "delayed_recall_labels.csv").exists()


def test_missing_required_file_fails(tmp_path: Path) -> None:
    directory = tmp_path / "evaluation"
    directory.mkdir()
    with pytest.raises(FileNotFoundError):
        load_evaluation_bundle(directory)
