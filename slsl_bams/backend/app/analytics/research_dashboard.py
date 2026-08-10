from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markupsafe import Markup

REQUIRED_SOURCE_FILES = (
    "scheduler_arm_summary.csv", "pairwise_retention_comparisons.csv",
    "delayed_recall_labels.csv", "evaluation_report.json",
)
OPTIONAL_SOURCE_FILES = ("hybrid_calibration.csv", "hybrid_prediction_labels.csv")


@dataclass(frozen=True)
class EvaluationBundle:
    source_directory: Path
    arm_summary: pd.DataFrame
    comparisons: pd.DataFrame
    delayed_labels: pd.DataFrame
    calibration: pd.DataFrame
    hybrid_predictions: pd.DataFrame
    evaluation_report: dict[str, Any]


@dataclass(frozen=True)
class DashboardBuildResult:
    output_directory: Path
    dashboard_path: Path
    summary_path: Path
    manifest_path: Path
    checksum_path: Path
    zip_path: Path


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: " + ", ".join(missing))


def load_evaluation_bundle(evaluation_directory: str | Path) -> EvaluationBundle:
    root = Path(evaluation_directory).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    for filename in REQUIRED_SOURCE_FILES:
        if not (root / filename).exists():
            raise FileNotFoundError(f"Required evaluation file was not found: {root / filename}")
    arms = pd.read_csv(root / "scheduler_arm_summary.csv")
    comparisons = pd.read_csv(root / "pairwise_retention_comparisons.csv")
    labels = pd.read_csv(root / "delayed_recall_labels.csv")
    calibration = (pd.read_csv(root / "hybrid_calibration.csv")
                   if (root / "hybrid_calibration.csv").exists() else pd.DataFrame())
    predictions = (pd.read_csv(root / "hybrid_prediction_labels.csv")
                   if (root / "hybrid_prediction_labels.csv").exists() else pd.DataFrame())
    report = json.loads((root / "evaluation_report.json").read_text(encoding="utf-8"))
    _require_columns(arms, {
        "arm", "learners", "delayed_labels", "retention_rate",
        "mean_quality_score", "mean_response_time_ms",
        "retained_recalls_per_100_events", "retained_recalls_per_response_hour",
    }, "scheduler_arm_summary.csv")
    _require_columns(comparisons, {
        "first_arm", "second_arm", "difference", "ci_95_low", "ci_95_high",
    }, "pairwise_retention_comparisons.csv")
    _require_columns(labels, {
        "student_id", "sign_id", "arm", "future_recall_success",
    }, "delayed_recall_labels.csv")
    return EvaluationBundle(root, arms, comparisons, labels, calibration, predictions, report)


def _chart(frame: pd.DataFrame, column: str, title: str, ylabel: str) -> str:
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    values = frame[column].astype(float)
    axis.bar(frame["arm"], values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _synthetic(bundle: EvaluationBundle) -> bool:
    if not bundle.hybrid_predictions.empty and "synthetic_model" in bundle.hybrid_predictions:
        return bool(bundle.hybrid_predictions["synthetic_model"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        ).any())
    return any("synthetic" in str(item).lower()
               for item in bundle.evaluation_report.get("limitations", []))


def build_quality_warnings(bundle: EvaluationBundle, *, minimum_learners_per_arm: int) -> list[str]:
    warnings = []
    missing = {"random_control", "modified_sm2", "hybrid_ml"} - set(bundle.arm_summary["arm"])
    if missing:
        warnings.append("Missing experimental arms: " + ", ".join(sorted(missing)) + ".")
    for row in bundle.arm_summary.itertuples():
        if int(row.learners) < minimum_learners_per_arm:
            warnings.append(f"{row.arm} has only {int(row.learners)} learners; the configured minimum is {minimum_learners_per_arm}.")
        if int(row.delayed_labels) < 30:
            warnings.append(f"{row.arm} has fewer than 30 delayed-recall labels.")
    if _synthetic(bundle):
        warnings.append("The Hybrid ML arm contains predictions from a synthetic-data model. These results are for software verification only.")
    if bundle.calibration.empty:
        warnings.append("No hybrid calibration table was available.")
    for row in bundle.comparisons.itertuples():
        if not pd.isna(row.ci_95_low) and float(row.ci_95_low) <= 0 <= float(row.ci_95_high):
            warnings.append(f"The comparison {row.first_arm} versus {row.second_arm} includes zero within its 95% interval.")
    contamination = bundle.delayed_labels.groupby("student_id")["arm"].nunique()
    if (contamination > 1).any():
        warnings.append(f"{int((contamination > 1).sum())} learners appear in more than one experimental arm.")
    return warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_research_dashboard(
    *, bundle: EvaluationBundle, template_path: str | Path,
    output_directory: str | Path, report_title: str,
    minimum_learners_per_arm: int, include_raw_labels: bool,
) -> DashboardBuildResult:
    output = Path(output_directory).resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    warnings = build_quality_warnings(bundle, minimum_learners_per_arm=minimum_learners_per_arm)
    synthetic = _synthetic(bundle)
    labels = bundle.delayed_labels
    total_learners = int(labels["student_id"].nunique())
    total_labels = len(labels)
    overall_retention = float(labels["future_recall_success"].mean()) if total_labels else 0.0
    arm_table = Markup(bundle.arm_summary.to_html(index=False, border=0, classes=["data-table"]))
    comparison_table = Markup(bundle.comparisons.to_html(index=False, border=0, classes=["data-table"]))
    charts = {
        "retention": _chart(bundle.arm_summary.assign(
            retention_percent=bundle.arm_summary["retention_rate"] * 100
        ), "retention_percent", "Delayed Receptive Retention", "Retention rate (%)"),
        "efficiency": _chart(bundle.arm_summary, "retained_recalls_per_100_events",
                             "Learning Efficiency", "Retained recalls per 100 events"),
        "labels": _chart(bundle.arm_summary, "delayed_labels",
                         "Delayed-Recall Evidence Volume", "Eligible labels"),
        "pairwise": None, "calibration": None,
    }
    template_file = Path(template_path).resolve()
    environment = Environment(
        loader=FileSystemLoader(str(template_file.parent)), undefined=StrictUndefined,
        autoescape=select_autoescape(["html"]),
    )
    html = environment.get_template(template_file.name).render(
        report_title=report_title,
        experiment_name=bundle.evaluation_report.get("experiment_name", "unknown"),
        total_learners=total_learners, total_labels=total_labels,
        overall_retention=overall_retention, warnings=warnings,
        ready_for_interpretation=not warnings and not synthetic,
        synthetic_model_used=synthetic, arm_table=arm_table,
        comparison_table=comparison_table, charts=charts,
        hybrid_metrics=bundle.evaluation_report.get("hybrid_prediction_metrics"),
        source_generated_at=bundle.evaluation_report.get("generated_at"),
        label_definition=bundle.evaluation_report.get("label_definition", {}),
    )
    dashboard = output / "research_dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    copied = []
    for filename in ["scheduler_arm_summary.csv", "pairwise_retention_comparisons.csv",
                     "evaluation_report.json", *OPTIONAL_SOURCE_FILES]:
        source = bundle.source_directory / filename
        if source.exists():
            target = data_dir / filename
            shutil.copy2(source, target)
            copied.append(target)
    if include_raw_labels:
        target = data_dir / "delayed_recall_labels.csv"
        shutil.copy2(bundle.source_directory / target.name, target)
        copied.append(target)
    summary = output / "research_summary.json"
    summary.write_text(json.dumps({
        "report_title": report_title, "experiment_name": bundle.evaluation_report.get("experiment_name"),
        "total_learners": total_learners, "total_labels": total_labels,
        "overall_retention": overall_retention, "synthetic_model_used": synthetic,
        "ready_for_interpretation": not warnings and not synthetic, "warnings": warnings,
        "raw_labels_included": include_raw_labels,
    }, indent=2), encoding="utf-8")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(bundle.source_directory),
        "files": [dashboard.name, summary.name] + [str(path.relative_to(output)) for path in copied],
    }, indent=2), encoding="utf-8")
    checksum = output / "SHA256SUMS.txt"
    package_files = [dashboard, summary, manifest, *copied]
    checksum.write_text("\n".join(f"{_sha256(path)}  {path.relative_to(output)}" for path in package_files) + "\n", encoding="utf-8")
    zip_base = output.parent / output.name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=output))
    return DashboardBuildResult(output, dashboard, summary, manifest, checksum, zip_path)
