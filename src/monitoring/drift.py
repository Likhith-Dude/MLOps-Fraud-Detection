"""Data and prediction drift detection using Evidently."""
from __future__ import annotations

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from pathlib import Path


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str | Path = "docs/drift_report.html",
) -> dict:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    share_drifted = result["metrics"][0]["result"]["share_of_drifted_columns"]
    return {"drift_detected": drift_detected, "share_drifted_columns": share_drifted}


def generate_model_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str | Path = "docs/model_report.html",
) -> None:
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(output_path))
