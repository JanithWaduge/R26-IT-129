from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.analytics.research_dashboard import build_research_dashboard, load_evaluation_bundle
from app.core.config import get_settings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SLSL-BAMS supervisor research dashboard.")
    parser.add_argument("--evaluation-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-raw-labels", action="store_true")
    return parser.parse_args()


def find_latest_evaluation(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(root)
    required_files = {
        "scheduler_arm_summary.csv",
        "pairwise_retention_comparisons.csv",
        "delayed_recall_labels.csv",
        "evaluation_report.json",
    }
    directories = [
        path for path in root.iterdir()
        if path.is_dir()
        and all((path / filename).exists() for filename in required_files)
    ]
    if not directories:
        raise FileNotFoundError(
            f"No complete evaluation directories were found under {root}. "
            "Run `python -m scripts.evaluate_scheduler_experiment` after the "
            "experiment has produced delayed-recall data, then retry."
        )
    return max(directories, key=lambda path: path.stat().st_mtime)


def main() -> None:
    arguments = parse_arguments()
    settings = get_settings()
    try:
        evaluation = (Path(arguments.evaluation_dir) if arguments.evaluation_dir
                      else find_latest_evaluation(Path(settings.experiment_output_directory)))
        bundle = load_evaluation_bundle(evaluation)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Cannot build research dashboard: {error}") from None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = bundle.evaluation_report.get("experiment_name", "scheduler-evaluation")
    output = (Path(arguments.output_dir) if arguments.output_dir else
              Path(settings.research_report_output_directory) / f"{experiment}_{timestamp}")
    result = build_research_dashboard(
        bundle=bundle, template_path=Path("templates/research_dashboard.html"),
        output_directory=output, report_title=settings.research_report_title,
        minimum_learners_per_arm=settings.experiment_min_learners_per_arm,
        include_raw_labels=(arguments.include_raw_labels
                            or settings.research_report_include_raw_labels),
    )
    print(f"Dashboard: {result.dashboard_path}")
    print(f"ZIP package: {result.zip_path}")
    print(f"Checksums: {result.checksum_path}")


if __name__ == "__main__":
    main()
