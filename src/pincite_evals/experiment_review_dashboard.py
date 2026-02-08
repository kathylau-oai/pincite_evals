"""Interactive dashboard for reviewing experiment and grader results."""

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_ROOT_DEFAULT = Path("results/experiments")
EXPERIMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
EXPERIMENT_KEY_COLUMNS = ["model_config", "source_row_index", "item_id"]
USER_QUERY_COLUMN_CANDIDATES = ["source_user_query", "user_query", "rendered_user_prompt"]
GRADER_COLUMN_PATTERN = re.compile(
    r"^grader_(?P<grader_name>.+)_(passed|status|score|label|reason|details_json)$"
)


def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "Space Grotesk", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(18, 151, 147, 0.12), transparent 34%),
        radial-gradient(circle at 87% 4%, rgba(29, 103, 198, 0.11), transparent 35%),
        linear-gradient(180deg, #f3f8ff 0%, #f8fbff 48%, #ffffff 100%);
}

.block-container {
    padding-top: 1.05rem;
    padding-bottom: 1.8rem;
}

.hero {
    background: linear-gradient(130deg, rgba(8, 123, 186, 0.16) 0%, rgba(31, 157, 123, 0.12) 100%);
    border: 1px solid rgba(8, 123, 186, 0.28);
    border-radius: 16px;
    padding: 0.95rem 1.05rem;
    margin-bottom: 0.7rem;
}

.hero h2 {
    margin: 0;
    font-size: 1.26rem;
    letter-spacing: 0.01em;
}

.hero p {
    margin: 0.24rem 0 0;
    color: #233544;
}

.metric-card {
    border-radius: 12px;
    border: 1px solid #cfe0f6;
    background: rgba(255, 255, 255, 0.95);
    padding: 0.65rem 0.78rem;
    min-height: 112px;
}

.metric-card.failure {
    border-color: #f1bbb6;
    background: rgba(255, 247, 246, 0.94);
}

.metric-card.success {
    border-color: #b9ddcf;
    background: rgba(244, 255, 248, 0.94);
}

.metric-title {
    font-size: 0.82rem;
    color: #49607a;
    margin: 0;
}

.metric-value {
    margin: 0.12rem 0 0;
    font-size: 1.38rem;
    font-weight: 700;
    color: #162537;
}

.metric-subtext {
    margin: 0.22rem 0 0;
    color: #516880;
    font-size: 0.8rem;
    line-height: 1.3;
}

.review-block {
    border: 1px solid #d7e3f4;
    border-radius: 12px;
    background: #ffffff;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.76rem;
}

.review-block h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.94rem;
    color: #1a2a3d;
}

.review-block pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 460px;
    overflow: auto;
    font-size: 0.82rem;
    line-height: 1.38;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

.status-strip {
    margin-bottom: 0.7rem;
    display: flex;
    gap: 0.44rem;
    flex-wrap: wrap;
}

.status-pill {
    border-radius: 999px;
    border: 1px solid #c8d9ef;
    background: #f5f9ff;
    color: #28425f;
    font-size: 0.76rem;
    padding: 0.16rem 0.54rem;
}

.status-pill.failure {
    border-color: #e5b2ad;
    background: #fff1ef;
    color: #7d2b22;
}

.status-pill.success {
    border-color: #b4d8c8;
    background: #effcf4;
    color: #205f3f;
}

[data-testid="stMarkdownContainer"] code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _parse_experiment_timestamp(run_directory_name: str) -> Optional[datetime]:
    timestamp_text = run_directory_name.rsplit("_", 2)
    if len(timestamp_text) < 3:
        return None
    run_timestamp = "_".join(timestamp_text[-2:])
    try:
        return datetime.strptime(run_timestamp, EXPERIMENT_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def list_experiment_runs(results_root: Path) -> List[Path]:
    if not results_root.exists():
        return []

    run_directories: List[Path] = []
    for run_directory in results_root.iterdir():
        if not run_directory.is_dir():
            continue
        predictions_file_path = run_directory / "final" / "predictions_with_grades.csv"
        if predictions_file_path.exists():
            run_directories.append(run_directory)

    def run_sort_key(run_path: Path) -> tuple[int, datetime]:
        parsed_timestamp = _parse_experiment_timestamp(run_path.name)
        if parsed_timestamp is not None:
            return (1, parsed_timestamp)
        return (0, datetime.fromtimestamp(run_path.stat().st_mtime))

    return sorted(run_directories, key=run_sort_key, reverse=True)


def _load_manifest_payload(experiment_path: Path) -> Dict[str, Any]:
    manifest_path = experiment_path / "manifest.json"
    if not manifest_path.exists():
        return {}

    manifest_text = manifest_path.read_text(encoding="utf-8").strip()
    if not manifest_text:
        return {}

    try:
        manifest_payload = json.loads(manifest_text)
    except json.JSONDecodeError:
        return {}

    return manifest_payload if isinstance(manifest_payload, dict) else {}


@st.cache_data(show_spinner=False)
def load_experiment_frames(experiment_directory: str) -> Dict[str, Any]:
    experiment_path = Path(experiment_directory)
    final_directory = experiment_path / "final"

    predictions_csv_path = final_directory / "predictions_with_grades.csv"
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {predictions_csv_path}")

    predictions_frame = pd.read_csv(predictions_csv_path)

    metrics_csv_path = final_directory / "metrics_summary.csv"
    metrics_frame = pd.read_csv(metrics_csv_path) if metrics_csv_path.exists() else pd.DataFrame()

    grader_metrics_csv_path = final_directory / "grader_metrics_summary.csv"
    grader_metrics_frame = pd.read_csv(grader_metrics_csv_path) if grader_metrics_csv_path.exists() else pd.DataFrame()

    errors_csv_path = final_directory / "errors.csv"
    errors_frame = pd.read_csv(errors_csv_path) if errors_csv_path.exists() else pd.DataFrame()

    return {
        "predictions": predictions_frame,
        "metrics": metrics_frame,
        "grader_metrics": grader_metrics_frame,
        "errors": errors_frame,
        "manifest": _load_manifest_payload(experiment_path),
    }


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    normalized_values: List[object] = []
    for value in series:
        if pd.isna(value):
            normalized_values.append(pd.NA)
            continue

        if isinstance(value, bool):
            normalized_values.append(value)
            continue

        if isinstance(value, (int, float)) and value in (0, 1):
            normalized_values.append(bool(int(value)))
            continue

        normalized_text = str(value).strip().lower()
        if normalized_text in {"true", "1", "yes"}:
            normalized_values.append(True)
        elif normalized_text in {"false", "0", "no"}:
            normalized_values.append(False)
        else:
            normalized_values.append(pd.NA)

    return pd.Series(normalized_values, index=series.index, dtype="boolean")


def _first_available_column(frame: pd.DataFrame, candidate_columns: List[str]) -> Optional[str]:
    for column_name in candidate_columns:
        if column_name in frame.columns:
            return column_name
    return None


def _format_display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    if pd.isna(value):
        return ""
    return str(value)


def _prepare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    display_frame = frame.copy()
    for column_name in display_frame.columns:
        if pd.api.types.is_object_dtype(display_frame[column_name]):
            display_frame[column_name] = display_frame[column_name].map(_format_display_value)
    return display_frame


def _render_text_block(title: str, body: str) -> None:
    safe_title = html.escape(title)
    safe_body = html.escape(body)
    st.markdown(
        f"""
<div class="review-block">
  <h4>{safe_title}</h4>
  <pre>{safe_body}</pre>
</div>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value: Any, precision: int = 1) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if pd.isna(numeric_value):
        return "n/a"

    return f"{numeric_value:.{precision}f}"


def _format_percent(value: Any) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if pd.isna(numeric_value):
        return "n/a"

    return f"{numeric_value * 100:.1f}%"


def _normalize_status_label(status_value: Any, passed_value: object) -> str:
    status_text = str(status_value).strip().lower()
    normalized_passed = _normalize_bool_series(pd.Series([passed_value])).iloc[0]

    if status_text:
        if status_text == "completed":
            if pd.isna(normalized_passed):
                return "COMPLETED"
            return "PASS" if bool(normalized_passed) else "FAIL"
        return status_text.upper()

    if pd.isna(normalized_passed):
        return "NOT_RUN"
    return "PASS" if bool(normalized_passed) else "FAIL"


def _status_css_class(status_text: str) -> str:
    normalized_status = status_text.strip().lower()
    if normalized_status in {"pass", "completed"}:
        return "success"
    if normalized_status in {"fail", "error", "incomplete", "skipped_model_error", "skipped_dry_run"}:
        return "failure"
    return ""


def _render_metric_card(title: str, value: str, subtext: str, tone: str = "") -> None:
    safe_title = html.escape(title)
    safe_value = html.escape(value)
    safe_subtext = html.escape(subtext)
    css_class = "metric-card"
    if tone:
        css_class = f"metric-card {tone}"

    st.markdown(
        f"""
<div class="{css_class}">
  <p class="metric-title">{safe_title}</p>
  <p class="metric-value">{safe_value}</p>
  <p class="metric-subtext">{safe_subtext}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def discover_grader_names(predictions_frame: pd.DataFrame) -> List[str]:
    grader_names = set()
    for column_name in predictions_frame.columns:
        column_match = GRADER_COLUMN_PATTERN.match(column_name)
        if column_match is not None:
            grader_names.add(column_match.group("grader_name"))
    return sorted(grader_names)


def add_failure_columns(predictions_frame: pd.DataFrame, grader_names: List[str]) -> pd.DataFrame:
    enriched_frame = predictions_frame.copy()

    response_status_series = enriched_frame.get("response_status", pd.Series(index=enriched_frame.index, dtype=object))
    enriched_frame["model_failed"] = response_status_series.fillna("missing") != "completed"

    has_template_grade = "grade_passed" in enriched_frame.columns
    grade_passed_series = _normalize_bool_series(
        enriched_frame.get("grade_passed", pd.Series(index=enriched_frame.index, dtype=object))
    )
    required_graders_series = _normalize_bool_series(
        enriched_frame.get("overall_required_graders_passed", pd.Series(index=enriched_frame.index, dtype=object))
    )

    enriched_frame["template_grade_available"] = has_template_grade
    enriched_frame["template_grade_failed"] = (
        grade_passed_series.fillna(True) == False if has_template_grade else pd.Series(False, index=enriched_frame.index)
    )
    enriched_frame["required_graders_failed"] = required_graders_series.fillna(True) == False

    any_grader_failed = pd.Series(False, index=enriched_frame.index)
    for grader_name in grader_names:
        grader_pass_column = f"grader_{grader_name}_passed"
        grader_status_column = f"grader_{grader_name}_status"

        if grader_pass_column in enriched_frame.columns:
            grader_passed_series = _normalize_bool_series(enriched_frame[grader_pass_column])
            any_grader_failed = any_grader_failed | (grader_passed_series.fillna(True) == False)

        if grader_status_column in enriched_frame.columns:
            grader_status_values = enriched_frame[grader_status_column].fillna("").astype(str).str.strip().str.lower()
            status_failures = grader_status_values.isin({"error", "incomplete", "skipped_model_error", "skipped_dry_run"})
            any_grader_failed = any_grader_failed | status_failures

    enriched_frame["any_grader_failed"] = any_grader_failed
    enriched_frame["has_any_failure"] = (
        enriched_frame["model_failed"]
        | enriched_frame["template_grade_failed"]
        | enriched_frame["required_graders_failed"]
        | enriched_frame["any_grader_failed"]
    )

    return enriched_frame


def _join_unique_values(series: pd.Series) -> str:
    values = sorted({str(value).strip() for value in series if pd.notna(value) and str(value).strip()})
    return ", ".join(values)


def merge_error_summary(predictions_frame: pd.DataFrame, errors_frame: pd.DataFrame) -> pd.DataFrame:
    if errors_frame.empty:
        enriched_frame = predictions_frame.copy()
        enriched_frame["error_row_count"] = 0
        enriched_frame["error_components"] = ""
        enriched_frame["error_statuses"] = ""
        enriched_frame["has_error_row"] = False
        return enriched_frame

    grouped_error_frame = (
        errors_frame.groupby(EXPERIMENT_KEY_COLUMNS, dropna=False)
        .agg(
            error_row_count=("status", "count"),
            error_components=("component_name", _join_unique_values),
            error_statuses=("status", _join_unique_values),
        )
        .reset_index()
    )

    enriched_frame = predictions_frame.merge(grouped_error_frame, on=EXPERIMENT_KEY_COLUMNS, how="left")
    enriched_frame["error_row_count"] = enriched_frame["error_row_count"].fillna(0).astype(int)
    enriched_frame["error_components"] = enriched_frame["error_components"].fillna("")
    enriched_frame["error_statuses"] = enriched_frame["error_statuses"].fillna("")
    enriched_frame["has_error_row"] = enriched_frame["error_row_count"] > 0
    return enriched_frame


def apply_filters(
    predictions_frame: pd.DataFrame,
    selected_model_configs: List[str],
    selected_error_modes: List[str],
    only_failures: bool,
    selected_failure_columns: List[str],
) -> pd.DataFrame:
    filtered_frame = predictions_frame.copy()

    if selected_model_configs:
        filtered_frame = filtered_frame[filtered_frame["model_config"].isin(selected_model_configs)]

    if selected_error_modes:
        filtered_frame = filtered_frame[filtered_frame["target_error_mode"].isin(selected_error_modes)]

    if only_failures:
        if selected_failure_columns:
            failure_mask = pd.Series(False, index=filtered_frame.index)
            for failure_column in selected_failure_columns:
                failure_mask = failure_mask | filtered_frame[failure_column].fillna(False)
            filtered_frame = filtered_frame[failure_mask]
        else:
            filtered_frame = filtered_frame[filtered_frame["has_any_failure"]]

    sort_columns = [column_name for column_name in ["source_row_index", "item_id"] if column_name in filtered_frame.columns]
    if sort_columns:
        filtered_frame = filtered_frame.sort_values(sort_columns, kind="mergesort")

    return filtered_frame.reset_index(drop=True)


def _build_grader_view(selected_row: pd.Series, grader_names: List[str]) -> pd.DataFrame:
    grader_rows: List[Dict[str, str]] = []
    for grader_name in grader_names:
        passed_column_name = f"grader_{grader_name}_passed"
        status_column_name = f"grader_{grader_name}_status"
        score_column_name = f"grader_{grader_name}_score"
        reason_column_name = f"grader_{grader_name}_reason"

        passed_value = selected_row.get(passed_column_name)
        status_value = selected_row.get(status_column_name)
        score_value = selected_row.get(score_column_name)
        detail_value = _format_display_value(selected_row.get(reason_column_name, ""))

        grader_status = _normalize_status_label(status_value=status_value, passed_value=passed_value)

        score_text = ""
        if pd.notna(score_value):
            score_text = f"{float(score_value):.3f}" if isinstance(score_value, (float, int)) else str(score_value)

        grader_rows.append(
            {
                "grader": grader_name,
                "status": grader_status,
                "score": score_text,
                "source": "predictions_with_grades",
                "detail": detail_value,
            }
        )

    overall_pass = _normalize_bool_series(pd.Series([selected_row.get("overall_required_graders_passed")])).iloc[0]
    overall_reason = _format_display_value(selected_row.get("overall_required_graders_reason", ""))
    if not grader_rows:
        if pd.isna(overall_pass):
            summary_status = "UNKNOWN"
        elif overall_pass:
            summary_status = "PASS"
        else:
            summary_status = "FAIL"
        grader_rows.append(
            {
                "grader": "required_graders_summary",
                "status": summary_status,
                "score": "",
                "source": "predictions_with_grades",
                "detail": overall_reason,
            }
        )
    elif overall_reason:
        grader_rows.append(
            {
                "grader": "required_graders_summary",
                "status": "PASS" if overall_pass == True else "FAIL" if overall_pass == False else "UNKNOWN",
                "score": "",
                "source": "predictions_with_grades",
                "detail": overall_reason,
            }
        )

    return pd.DataFrame(grader_rows)


def _build_error_component_view(selected_row_errors: pd.DataFrame) -> pd.DataFrame:
    if selected_row_errors.empty:
        return pd.DataFrame(columns=["grader", "status", "source", "detail"])

    error_rows: List[Dict[str, str]] = []
    for _, error_row in selected_row_errors.iterrows():
        detail_text = _format_display_value(error_row.get("error_message"))
        incomplete_reason = _format_display_value(error_row.get("incomplete_reason"))
        if incomplete_reason:
            detail_text = f"{detail_text} | incomplete_reason={incomplete_reason}" if detail_text else incomplete_reason

        error_rows.append(
            {
                "grader": _format_display_value(error_row.get("component_name")),
                "status": _format_display_value(error_row.get("status")).upper(),
                "source": "errors.csv",
                "detail": detail_text,
            }
        )
    return pd.DataFrame(error_rows)


def _build_grader_pass_rate_frame(predictions_frame: pd.DataFrame, grader_names: List[str]) -> pd.DataFrame:
    grader_rows: List[Dict[str, object]] = []
    for grader_name in grader_names:
        passed_column_name = f"grader_{grader_name}_passed"
        if passed_column_name not in predictions_frame.columns:
            continue

        passed_series = _normalize_bool_series(predictions_frame[passed_column_name])
        valid_rows = predictions_frame[passed_series.notna()].copy()
        valid_rows["grader_passed"] = passed_series[passed_series.notna()].astype(bool).values
        if valid_rows.empty:
            continue

        grouped = (
            valid_rows.groupby("target_error_mode", dropna=False)["grader_passed"]
            .agg(evaluated_rows="count", pass_rate="mean")
            .reset_index()
        )

        grouped["grader_name"] = grader_name
        grader_rows.append(grouped)

    if not grader_rows:
        return pd.DataFrame()

    grader_pass_rate_frame = pd.concat(grader_rows, ignore_index=True)
    grader_pass_rate_frame["pass_rate_percent"] = grader_pass_rate_frame["pass_rate"] * 100.0
    return grader_pass_rate_frame


def _build_model_summary_frame(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty:
        return pd.DataFrame(
            columns=[
                "model_config",
                "rows",
                "failure_rows",
                "failure_rate",
                "non_completed_rows",
                "required_grader_pass_rate",
            ]
        )

    summary_rows: List[Dict[str, Any]] = []
    grouped = predictions_frame.groupby("model_config", dropna=False)
    for model_config, model_rows in grouped:
        row_count = int(model_rows.shape[0])
        failure_rows = int(model_rows["has_any_failure"].fillna(False).sum())
        non_completed_rows = int((model_rows["response_status"].fillna("missing") != "completed").sum())
        required_pass = _normalize_bool_series(model_rows.get("overall_required_graders_passed", pd.Series(dtype=object)))
        pass_rate = float(required_pass.dropna().mean()) if not required_pass.dropna().empty else None

        summary_rows.append(
            {
                "model_config": _format_display_value(model_config),
                "rows": row_count,
                "failure_rows": failure_rows,
                "failure_rate": (failure_rows / row_count) if row_count else 0.0,
                "non_completed_rows": non_completed_rows,
                "required_grader_pass_rate": pass_rate,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("model_config", kind="mergesort").reset_index(drop=True)


def _build_mode_summary_frame(predictions_frame: pd.DataFrame) -> pd.DataFrame:
    if predictions_frame.empty:
        return pd.DataFrame(columns=["target_error_mode", "rows", "failure_rows", "failure_rate"])

    grouped = (
        predictions_frame.groupby("target_error_mode", dropna=False)
        .agg(rows=("item_id", "count"), failure_rows=("has_any_failure", "sum"))
        .reset_index()
    )
    grouped["failure_rate"] = grouped["failure_rows"] / grouped["rows"].clip(lower=1)
    return grouped.sort_values("target_error_mode", kind="mergesort").reset_index(drop=True)


def _build_top_failure_reason_frame(predictions_frame: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if predictions_frame.empty or "overall_required_graders_reason" not in predictions_frame.columns:
        return pd.DataFrame(columns=["reason", "count"])

    reason_series = (
        predictions_frame["overall_required_graders_reason"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    reason_series = reason_series[
        (reason_series != "") & (reason_series.str.lower() != "all required graders passed.")
    ]
    if reason_series.empty:
        return pd.DataFrame(columns=["reason", "count"])

    reason_counts = reason_series.value_counts().head(top_n).reset_index()
    reason_counts.columns = ["reason", "count"]
    return reason_counts


def _build_search_mask(
    review_frame: pd.DataFrame,
    search_text: str,
    user_query_column_name: Optional[str],
) -> pd.Series:
    lower_search_text = search_text.strip().lower()
    if not lower_search_text:
        return pd.Series(True, index=review_frame.index)

    query_values = (
        review_frame[user_query_column_name].map(_format_display_value).str.lower()
        if user_query_column_name is not None
        else pd.Series("", index=review_frame.index)
    )

    return (
        review_frame["item_id"].map(_format_display_value).str.lower().str.contains(lower_search_text, regex=False)
        | review_frame.get("query_id", pd.Series("", index=review_frame.index))
        .map(_format_display_value)
        .str.lower()
        .str.contains(lower_search_text, regex=False)
        | review_frame["packet_id"].map(_format_display_value).str.lower().str.contains(lower_search_text, regex=False)
        | review_frame.get("target_error_mode", pd.Series("", index=review_frame.index))
        .map(_format_display_value)
        .str.lower()
        .str.contains(lower_search_text, regex=False)
        | query_values.str.contains(lower_search_text, regex=False)
    )


def _build_review_row_label(row: pd.Series) -> str:
    row_index_text = _format_display_value(row.get("source_row_index")) or "?"
    item_id_text = _format_display_value(row.get("item_id")) or "unknown_item"
    mode_text = _format_display_value(row.get("target_error_mode")) or "?"
    model_text = _format_display_value(row.get("model_config")) or "?"
    response_text = _format_display_value(row.get("response_status")) or "missing"
    return f"row {row_index_text} | {item_id_text} | mode {mode_text} | {model_text} | {response_text}"


def _extract_selected_row_errors(errors_frame: pd.DataFrame, selected_row: pd.Series) -> pd.DataFrame:
    if errors_frame.empty:
        return pd.DataFrame()

    return errors_frame[
        (errors_frame["model_config"] == selected_row.get("model_config"))
        & (errors_frame["source_row_index"] == selected_row.get("source_row_index"))
        & (errors_frame["item_id"] == selected_row.get("item_id"))
    ].copy()


def _style_figure(figure) -> None:
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=18, r=18, t=56, b=18),
        legend_title_text="",
        height=390,
    )


def _render_run_header(selected_run_directory: Path, manifest_payload: Dict[str, Any]) -> None:
    run_created_text = _format_display_value(manifest_payload.get("created_at"))
    artifact_level = _format_display_value(manifest_payload.get("artifact_level"))
    summary = manifest_payload.get("summary") if isinstance(manifest_payload.get("summary"), dict) else {}

    st.markdown(
        """
<div class="hero">
  <h2>Pincite Experiment Review Console</h2>
  <p>Fast failure triage with model/grader comparisons and row-level evidence in one place.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if manifest_payload:
        created_label = run_created_text or "unknown"
        artifact_label = artifact_level or "unknown"
        item_count = _format_display_value(summary.get("item_count", ""))
        completed_count = _format_display_value(summary.get("completed_item_count", ""))
        st.caption(
            " | ".join(
                [
                    f"Run folder: `{selected_run_directory.name}`",
                    f"Created: `{created_label}`",
                    f"Artifact level: `{artifact_label}`",
                    f"Items: `{item_count}`",
                    f"Completed: `{completed_count}`",
                ]
            )
        )
    else:
        st.caption(f"Run folder: `{selected_run_directory.name}`")


def render_dashboard() -> None:
    st.set_page_config(page_title="Pincite Experiment Review", page_icon=":bar_chart:", layout="wide")
    _inject_css()

    results_root = Path(
        st.sidebar.text_input("Results root directory", value=str(RESULTS_ROOT_DEFAULT), help="Folder with run outputs.")
    )

    run_directories = list_experiment_runs(results_root)
    if not run_directories:
        st.error(f"No experiment runs found under `{results_root}`.")
        return

    run_names = [run_directory.name for run_directory in run_directories]
    selected_run_name = st.sidebar.selectbox("Experiment run", options=run_names, index=0)
    selected_run_directory = run_directories[run_names.index(selected_run_name)]

    st.sidebar.caption(f"Viewing: `{selected_run_directory}`")

    frames = load_experiment_frames(str(selected_run_directory))
    predictions_frame = frames["predictions"]
    metrics_frame = frames["metrics"]
    grader_metrics_frame = frames["grader_metrics"]
    errors_frame = frames["errors"]
    manifest_payload = frames["manifest"]

    _render_run_header(selected_run_directory, manifest_payload)

    grader_names = discover_grader_names(predictions_frame)
    predictions_frame = add_failure_columns(predictions_frame, grader_names)
    predictions_frame = merge_error_summary(predictions_frame, errors_frame)

    available_model_configs = sorted(predictions_frame["model_config"].dropna().unique().tolist())
    selected_model_configs = st.sidebar.multiselect(
        "Model config",
        options=available_model_configs,
        default=available_model_configs,
    )

    available_error_modes = sorted(predictions_frame["target_error_mode"].dropna().unique().tolist())
    selected_error_modes = st.sidebar.multiselect(
        "Target error mode",
        options=available_error_modes,
        default=available_error_modes,
    )

    only_failures = st.sidebar.checkbox("Show only failures", value=True)

    failure_filter_options = {
        "Model request failures": "model_failed",
        "Required grader failures": "required_graders_failed",
        "Any grader failures": "any_grader_failed",
        "Rows with error logs": "has_error_row",
    }
    if not predictions_frame.empty and bool(predictions_frame["template_grade_available"].any()):
        failure_filter_options["Template grade failures"] = "template_grade_failed"

    selected_failure_labels = st.sidebar.multiselect(
        "Failure filters",
        options=list(failure_filter_options.keys()),
        default=list(failure_filter_options.keys()),
    )
    selected_failure_columns = [failure_filter_options[label] for label in selected_failure_labels]

    filtered_predictions_frame = apply_filters(
        predictions_frame=predictions_frame,
        selected_model_configs=selected_model_configs,
        selected_error_modes=selected_error_modes,
        only_failures=only_failures,
        selected_failure_columns=selected_failure_columns,
    )

    user_query_column_name = _first_available_column(predictions_frame, USER_QUERY_COLUMN_CANDIDATES)

    filtered_rows_count = int(filtered_predictions_frame.shape[0])
    total_rows_count = int(predictions_frame.shape[0])
    filtered_failure_count = int(filtered_predictions_frame["has_any_failure"].fillna(False).sum())
    filtered_non_completed_count = int(
        (filtered_predictions_frame["response_status"].fillna("missing") != "completed").sum()
    )
    filtered_error_log_count = int(filtered_predictions_frame["has_error_row"].fillna(False).sum())

    filtered_overall_pass = _normalize_bool_series(
        filtered_predictions_frame.get("overall_required_graders_passed", pd.Series(dtype=object))
    )
    filtered_overall_pass_rate = (
        float(filtered_overall_pass.dropna().mean()) if not filtered_overall_pass.dropna().empty else None
    )

    metric_columns = st.columns(5)
    with metric_columns[0]:
        _render_metric_card("Total rows", f"{total_rows_count:,}", "Rows in selected run")
    with metric_columns[1]:
        _render_metric_card("Filtered rows", f"{filtered_rows_count:,}", "Rows after current filters")
    with metric_columns[2]:
        failure_rate = (filtered_failure_count / filtered_rows_count) if filtered_rows_count else 0.0
        _render_metric_card(
            "Failure rate (filtered)",
            _format_percent(failure_rate),
            f"{filtered_failure_count:,} failing rows",
            tone="failure" if failure_rate > 0 else "success",
        )
    with metric_columns[3]:
        _render_metric_card(
            "Non-completed (filtered)",
            f"{filtered_non_completed_count:,}",
            "Model calls not in completed status",
            tone="failure" if filtered_non_completed_count > 0 else "success",
        )
    with metric_columns[4]:
        _render_metric_card(
            "Required grader pass",
            _format_percent(filtered_overall_pass_rate),
            f"{filtered_error_log_count:,} rows with error logs",
            tone="success" if (filtered_overall_pass_rate or 0) >= 0.8 else "",
        )

    if not metrics_frame.empty:
        summary = metrics_frame.iloc[0]
        st.caption(
            "Run summary: "
            f"graded pass rate={_format_percent(summary.get('graded_pass_rate'))}, "
            f"avg latency={_format_number(summary.get('latency_seconds_avg'), precision=2)}s, "
            f"avg total tokens={_format_number(summary.get('total_tokens_avg'), precision=0)}"
        )

    overview_tab, row_review_tab, charts_tab, raw_tab = st.tabs(
        ["Overview", "Row Review", "Comparisons", "Raw Data"]
    )

    with overview_tab:
        summary_col_1, summary_col_2 = st.columns([1.05, 0.95])

        with summary_col_1:
            st.subheader("Model summary (filtered rows)")
            model_summary_frame = _build_model_summary_frame(filtered_predictions_frame)
            if model_summary_frame.empty:
                st.info("No rows available for model summary.")
            else:
                display_frame = model_summary_frame.copy()
                display_frame["failure_rate"] = display_frame["failure_rate"].map(_format_percent)
                display_frame["required_grader_pass_rate"] = display_frame["required_grader_pass_rate"].map(_format_percent)
                st.dataframe(_prepare_display_frame(display_frame), width="stretch", hide_index=True, height=260)

        with summary_col_2:
            st.subheader("Mode summary (filtered rows)")
            mode_summary_frame = _build_mode_summary_frame(filtered_predictions_frame)
            if mode_summary_frame.empty:
                st.info("No rows available for mode summary.")
            else:
                mode_display_frame = mode_summary_frame.copy()
                mode_display_frame["failure_rate"] = mode_display_frame["failure_rate"].map(_format_percent)
                st.dataframe(_prepare_display_frame(mode_display_frame), width="stretch", hide_index=True, height=260)

        reason_frame = _build_top_failure_reason_frame(filtered_predictions_frame)
        st.subheader("Top failure reasons")
        if reason_frame.empty:
            st.caption("No failing reason text available in filtered rows.")
        else:
            st.dataframe(_prepare_display_frame(reason_frame), width="stretch", hide_index=True, height=220)

        st.subheader("Filtered experiment rows")
        if filtered_predictions_frame.empty:
            st.info("No rows match the selected filters.")
        else:
            preview_columns = [
                "source_row_index",
                "item_id",
                "packet_id",
                "target_error_mode",
                "model_config",
                "response_status",
                "overall_required_graders_passed",
                "overall_required_graders_reason",
                "model_failed",
                "any_grader_failed",
                "has_error_row",
                "latency_seconds",
                "total_tokens",
            ]
            if "grade_passed" in filtered_predictions_frame.columns:
                preview_columns.insert(6, "grade_passed")

            available_preview_columns = [
                column_name for column_name in preview_columns if column_name in filtered_predictions_frame.columns
            ]
            st.dataframe(
                _prepare_display_frame(filtered_predictions_frame[available_preview_columns]),
                width="stretch",
                hide_index=True,
                height=360,
            )

    with row_review_tab:
        st.subheader("Row-level review")
        if filtered_predictions_frame.empty:
            st.info("No rows available for drilldown with current filters.")
        else:
            search_text = st.text_input(
                "Search rows",
                value="",
                placeholder="Filter by item_id, query_id, packet_id, mode, or query text",
            )

            review_mask = _build_search_mask(
                review_frame=filtered_predictions_frame,
                search_text=search_text,
                user_query_column_name=user_query_column_name,
            )
            review_frame = filtered_predictions_frame[review_mask].reset_index(drop=True)

            if review_frame.empty:
                st.warning("No rows matched the search text with current filters.")
            else:
                row_labels = review_frame.apply(_build_review_row_label, axis=1).tolist()
                selected_row_label = st.selectbox("Select row", options=row_labels, index=0)
                selected_position = row_labels.index(selected_row_label)
                selected_row = review_frame.iloc[selected_position]

                response_status_text = _format_display_value(selected_row.get("response_status")) or "unknown"
                required_graders_passed = _normalize_bool_series(
                    pd.Series([selected_row.get("overall_required_graders_passed")])
                ).iloc[0]
                required_status = (
                    "PASS"
                    if required_graders_passed is True
                    else "FAIL"
                    if required_graders_passed is False
                    else "UNKNOWN"
                )
                mode_text = _format_display_value(selected_row.get("target_error_mode")) or "?"
                model_text = _format_display_value(selected_row.get("model_config")) or "?"

                # Show a compact status strip before detailed row text to speed triage.
                status_pills = [
                    (f"response={response_status_text}", _status_css_class(response_status_text)),
                    (f"required_graders={required_status}", _status_css_class(required_status)),
                    (f"mode={mode_text}", ""),
                    (f"model={model_text}", ""),
                ]
                pill_markup = "".join(
                    [
                        f'<span class="status-pill {pill_css}">{html.escape(pill_text)}</span>'
                        for pill_text, pill_css in status_pills
                    ]
                )
                st.markdown(f'<div class="status-strip">{pill_markup}</div>', unsafe_allow_html=True)

                detail_col_1, detail_col_2 = st.columns([1.15, 0.85])

                with detail_col_1:
                    query_text = (
                        _format_display_value(selected_row.get(user_query_column_name))
                        if user_query_column_name is not None
                        else ""
                    )
                    output_text = _format_display_value(selected_row.get("model_output", ""))
                    _render_text_block("User query", query_text or "[No user query found in this run schema]")
                    _render_text_block("Model output", output_text or "[No model output captured]")

                with detail_col_2:
                    st.markdown("**Row metadata**")
                    metadata_rows = pd.DataFrame(
                        [
                            {"field": "source_row_index", "value": selected_row.get("source_row_index")},
                            {"field": "item_id", "value": selected_row.get("item_id")},
                            {"field": "packet_id", "value": selected_row.get("packet_id")},
                            {"field": "query_id", "value": selected_row.get("query_id")},
                            {"field": "as_of_date", "value": selected_row.get("as_of_date")},
                            {"field": "target_error_mode", "value": selected_row.get("target_error_mode")},
                            {"field": "response_status", "value": selected_row.get("response_status")},
                            {"field": "overall_required_graders_passed", "value": selected_row.get("overall_required_graders_passed")},
                            {"field": "overall_required_graders_reason", "value": selected_row.get("overall_required_graders_reason")},
                            {"field": "latency_seconds", "value": selected_row.get("latency_seconds")},
                            {"field": "total_tokens", "value": selected_row.get("total_tokens")},
                        ]
                    )
                    if "grade_passed" in review_frame.columns:
                        grade_row = pd.DataFrame([{"field": "grade_passed", "value": selected_row.get("grade_passed")}])
                        metadata_rows = pd.concat([metadata_rows.iloc[:7], grade_row, metadata_rows.iloc[7:]], ignore_index=True)

                    st.dataframe(_prepare_display_frame(metadata_rows), width="stretch", hide_index=True, height=332)

                    selected_row_errors = _extract_selected_row_errors(errors_frame, selected_row)

                    st.markdown("**Grader outcomes**")
                    grader_view_frame = _build_grader_view(
                        selected_row=selected_row,
                        grader_names=grader_names,
                    )
                    error_component_view = _build_error_component_view(selected_row_errors)
                    combined_grader_view = pd.concat([grader_view_frame, error_component_view], ignore_index=True)
                    st.dataframe(_prepare_display_frame(combined_grader_view), width="stretch", hide_index=True, height=250)

                st.markdown("**Error log entries for this row**")
                selected_row_errors = _extract_selected_row_errors(errors_frame, selected_row)
                if errors_frame.empty:
                    st.caption("No `errors.csv` file in this run.")
                elif selected_row_errors.empty:
                    st.caption("No error rows for this item.")
                else:
                    st.dataframe(_prepare_display_frame(selected_row_errors), width="stretch", hide_index=True, height=220)

                with st.expander("Show full selected-row payload"):
                    st.json(json.loads(pd.Series(selected_row).to_json(force_ascii=True)))

    with charts_tab:
        st.subheader("Comparative plots")
        if filtered_predictions_frame.empty:
            st.info("No rows available for plots with current filters.")
        else:
            status_by_model_frame = (
                filtered_predictions_frame.groupby(["model_config", "response_status"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            status_by_model_frame["response_status"] = status_by_model_frame["response_status"].fillna("missing")
            status_by_model_figure = px.bar(
                status_by_model_frame,
                x="model_config",
                y="count",
                color="response_status",
                barmode="stack",
                title="Response status by model config",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            status_by_model_figure.update_layout(xaxis_title="Model config", yaxis_title="Rows")
            _style_figure(status_by_model_figure)
            st.plotly_chart(status_by_model_figure, width="stretch")

            failure_melt_frame = filtered_predictions_frame.melt(
                id_vars=["target_error_mode", "model_config"],
                value_vars=["model_failed", "template_grade_failed", "required_graders_failed", "any_grader_failed"],
                var_name="failure_type",
                value_name="is_failure",
            )
            failure_melt_frame = failure_melt_frame[failure_melt_frame["is_failure"] == True]
            failure_melt_frame["failure_type"] = failure_melt_frame["failure_type"].replace(
                {
                    "model_failed": "model_request",
                    "template_grade_failed": "template_grade",
                    "required_graders_failed": "required_graders",
                    "any_grader_failed": "any_grader",
                }
            )

            if not failure_melt_frame.empty:
                failure_mode_counts_frame = (
                    failure_melt_frame.groupby(["target_error_mode", "failure_type"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                failure_figure = px.bar(
                    failure_mode_counts_frame,
                    x="target_error_mode",
                    y="count",
                    color="failure_type",
                    barmode="group",
                    title="Failure counts by mode and type",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                failure_figure.update_layout(xaxis_title="Target error mode", yaxis_title="Failed rows")
                _style_figure(failure_figure)
                st.plotly_chart(failure_figure, width="stretch")

            grader_pass_rate_frame = _build_grader_pass_rate_frame(filtered_predictions_frame, grader_names)
            if not grader_pass_rate_frame.empty:
                grader_heatmap_frame = grader_pass_rate_frame.pivot_table(
                    index="grader_name",
                    columns="target_error_mode",
                    values="pass_rate_percent",
                    aggfunc="mean",
                )
                grader_heatmap = px.imshow(
                    grader_heatmap_frame,
                    text_auto=".1f",
                    color_continuous_scale="YlGnBu",
                    zmin=0,
                    zmax=100,
                    aspect="auto",
                    title="Grader pass rate heatmap (%)",
                    labels={"color": "Pass rate (%)"},
                )
                _style_figure(grader_heatmap)
                st.plotly_chart(grader_heatmap, width="stretch")

            if "latency_seconds" in filtered_predictions_frame.columns:
                latency_frame = filtered_predictions_frame.dropna(subset=["latency_seconds"]).copy()
                if not latency_frame.empty:
                    latency_figure = px.box(
                        latency_frame,
                        x="model_config",
                        y="latency_seconds",
                        color="target_error_mode",
                        points="outliers",
                        title="Latency distribution by model and mode",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                    )
                    latency_figure.update_layout(
                        xaxis_title="Model config",
                        yaxis_title="Latency (seconds)",
                    )
                    _style_figure(latency_figure)
                    st.plotly_chart(latency_figure, width="stretch")

            if "total_tokens" in filtered_predictions_frame.columns:
                token_frame = filtered_predictions_frame.dropna(subset=["total_tokens"]).copy()
                if not token_frame.empty:
                    token_figure = px.box(
                        token_frame,
                        x="model_config",
                        y="total_tokens",
                        color="target_error_mode",
                        points="outliers",
                        title="Total token distribution by model and mode",
                        color_discrete_sequence=px.colors.qualitative.Prism,
                    )
                    token_figure.update_layout(xaxis_title="Model config", yaxis_title="Total tokens")
                    _style_figure(token_figure)
                    st.plotly_chart(token_figure, width="stretch")

    with raw_tab:
        st.subheader("Raw tables and downloads")

        if not filtered_predictions_frame.empty:
            download_frame = _prepare_display_frame(filtered_predictions_frame)
            st.download_button(
                label="Download filtered rows CSV",
                data=download_frame.to_csv(index=False),
                file_name=f"{selected_run_name}_filtered_predictions.csv",
                mime="text/csv",
            )

        table_choice = st.selectbox(
            "Choose table",
            options=[
                "filtered_predictions",
                "metrics_summary",
                "grader_metrics_summary",
                "errors",
            ],
            index=0,
        )

        if table_choice == "filtered_predictions":
            st.dataframe(_prepare_display_frame(filtered_predictions_frame), width="stretch", hide_index=True, height=420)
        elif table_choice == "metrics_summary":
            st.dataframe(_prepare_display_frame(metrics_frame), width="stretch", hide_index=True, height=360)
        elif table_choice == "grader_metrics_summary":
            st.dataframe(_prepare_display_frame(grader_metrics_frame), width="stretch", hide_index=True, height=360)
        else:
            st.dataframe(_prepare_display_frame(errors_frame), width="stretch", hide_index=True, height=360)


def main() -> None:
    render_dashboard()


def launch() -> None:
    dashboard_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    from streamlit.web import cli as streamlit_cli

    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
