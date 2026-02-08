"""Interactive dashboard for reviewing experiment and grader results."""

import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_ROOT_DEFAULT = Path("results/experiments")
EXPERIMENT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
EXPERIMENT_KEY_COLUMNS = ["model_config", "source_row_index", "item_id"]
USER_QUERY_COLUMN_CANDIDATES = ["source_user_query", "user_query", "rendered_user_prompt"]


def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "Space Grotesk", sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f4f8ff 0%, #f2fbff 42%, #ffffff 100%);
}

.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, rgba(10, 116, 218, 0.12) 0%, rgba(4, 166, 168, 0.08) 100%);
    border: 1px solid rgba(10, 116, 218, 0.22);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
}

.hero h2 {
    margin: 0;
    font-size: 1.25rem;
}

.hero p {
    margin: 0.3rem 0 0;
    color: #1d2a38;
}

.review-block {
    border: 1px solid #d7e3f4;
    border-radius: 12px;
    background: #ffffff;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.8rem;
}

.review-block h4 {
    margin: 0 0 0.55rem 0;
    font-size: 0.95rem;
    color: #1a2a3d;
}

.review-block pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 420px;
    overflow: auto;
    font-size: 0.83rem;
    line-height: 1.35;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

[data-testid="stMarkdownContainer"] code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem;
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

    run_directories = [path for path in results_root.iterdir() if path.is_dir()]

    def run_sort_key(run_path: Path) -> tuple[int, datetime]:
        parsed_timestamp = _parse_experiment_timestamp(run_path.name)
        if parsed_timestamp is not None:
            return (1, parsed_timestamp)
        return (0, datetime.fromtimestamp(run_path.stat().st_mtime))

    return sorted(run_directories, key=run_sort_key, reverse=True)


@st.cache_data(show_spinner=False)
def load_experiment_frames(experiment_directory: str) -> Dict[str, pd.DataFrame]:
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


def discover_grader_names(predictions_frame: pd.DataFrame) -> List[str]:
    grader_names: List[str] = []
    for column_name in predictions_frame.columns:
        if column_name.startswith("grader_") and column_name.endswith("_passed"):
            grader_names.append(column_name[len("grader_") : -len("_passed")])
    return sorted(grader_names)


def add_failure_columns(predictions_frame: pd.DataFrame, grader_names: List[str]) -> pd.DataFrame:
    enriched_frame = predictions_frame.copy()

    response_status_series = enriched_frame.get("response_status", pd.Series(index=enriched_frame.index, dtype=object))
    enriched_frame["model_failed"] = response_status_series.fillna("missing") != "completed"

    grade_passed_series = _normalize_bool_series(
        enriched_frame.get("grade_passed", pd.Series(index=enriched_frame.index, dtype=object))
    )
    required_graders_series = _normalize_bool_series(
        enriched_frame.get("overall_required_graders_passed", pd.Series(index=enriched_frame.index, dtype=object))
    )

    enriched_frame["template_grade_failed"] = grade_passed_series.fillna(True) == False
    enriched_frame["required_graders_failed"] = required_graders_series.fillna(True) == False

    any_grader_failed = pd.Series(False, index=enriched_frame.index)
    for grader_name in grader_names:
        grader_column = f"grader_{grader_name}_passed"
        if grader_column not in enriched_frame.columns:
            continue
        grader_passed_series = _normalize_bool_series(enriched_frame[grader_column])
        any_grader_failed = any_grader_failed | (grader_passed_series.fillna(True) == False)

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

    return filtered_frame.sort_values(["source_row_index", "item_id"], kind="mergesort").reset_index(drop=True)


def _build_grader_view(selected_row: pd.Series, grader_names: List[str]) -> pd.DataFrame:
    grader_rows: List[Dict[str, str]] = []
    for grader_name in grader_names:
        passed_column_name = f"grader_{grader_name}_passed"
        score_column_name = f"grader_{grader_name}_score"

        passed_value = selected_row.get(passed_column_name)
        score_value = selected_row.get(score_column_name)
        normalized_passed_value = _normalize_bool_series(pd.Series([passed_value])).iloc[0]

        if pd.isna(normalized_passed_value):
            grader_status = "NOT_RUN"
        elif normalized_passed_value:
            grader_status = "PASS"
        else:
            grader_status = "FAIL"

        score_text = ""
        if pd.notna(score_value):
            score_text = f"{float(score_value):.3f}" if isinstance(score_value, (float, int)) else str(score_value)

        grader_rows.append(
            {
                "grader": grader_name,
                "status": grader_status,
                "score": score_text,
                "source": "predictions_with_grades",
                "detail": "",
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
        error_rows.append(
            {
                "grader": _format_display_value(error_row.get("component_name")),
                "status": _format_display_value(error_row.get("status")).upper(),
                "source": "errors.csv",
                "detail": _format_display_value(error_row.get("error_message")),
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


def render_dashboard() -> None:
    st.set_page_config(page_title="Pincite Experiment Review", page_icon=":bar_chart:", layout="wide")
    _inject_css()

    st.markdown(
        """
<div class="hero">
  <h2>Pincite Experiment Review Console</h2>
  <p>Inspect failures quickly, compare grader outcomes, and spot regressions by error mode.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

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
        "Template grade failures": "template_grade_failed",
        "Required grader failures": "required_graders_failed",
        "Any grader failures": "any_grader_failed",
        "Rows with error logs": "has_error_row",
    }
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

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Total rows", f"{predictions_frame.shape[0]:,}")
    metric_col_2.metric("Filtered rows", f"{filtered_predictions_frame.shape[0]:,}")
    metric_col_3.metric("Failure rows", f"{int(predictions_frame['has_any_failure'].sum()):,}")
    metric_col_4.metric("Error log rows", f"{int(predictions_frame['has_error_row'].sum()):,}")

    if not metrics_frame.empty:
        summary = metrics_frame.iloc[0]
        st.caption(
            "Run summary: "
            f"graded pass rate={summary.get('graded_pass_rate', 0):.1%}, "
            f"avg latency={summary.get('latency_seconds_avg', 0):.2f}s, "
            f"avg total tokens={summary.get('total_tokens_avg', 0):,.0f}"
        )

    overview_tab, failures_tab, plots_tab, raw_tab = st.tabs(["Overview", "Failure Review", "Plots", "Raw Tables"])

    with overview_tab:
        preview_columns = [
            "source_row_index",
            "item_id",
            "packet_id",
            "target_error_mode",
            "model_config",
            "response_status",
            "grade_passed",
            "overall_required_graders_passed",
            "model_failed",
            "any_grader_failed",
            "has_error_row",
            "latency_seconds",
            "total_tokens",
        ]
        available_preview_columns = [column for column in preview_columns if column in filtered_predictions_frame.columns]

        st.subheader("Filtered experiment rows")
        if filtered_predictions_frame.empty:
            st.info("No rows match the selected filters.")
        else:
            st.dataframe(
                _prepare_display_frame(filtered_predictions_frame[available_preview_columns]),
                width="stretch",
                hide_index=True,
                height=360,
            )

    with failures_tab:
        st.subheader("Failure drilldown")
        if filtered_predictions_frame.empty:
            st.info("No rows available for drilldown with current filters.")
        else:
            search_text = st.text_input(
                "Search rows",
                value="",
                placeholder="Filter by item_id, query_id, packet_id, or query text",
            ).strip().lower()

            review_frame = filtered_predictions_frame.copy()
            if search_text:
                query_values = (
                    review_frame[user_query_column_name].map(_format_display_value).str.lower()
                    if user_query_column_name is not None
                    else pd.Series("", index=review_frame.index)
                )
                review_mask = (
                    review_frame["item_id"].map(_format_display_value).str.lower().str.contains(search_text, regex=False)
                    | review_frame.get("query_id", pd.Series("", index=review_frame.index))
                    .map(_format_display_value)
                    .str.lower()
                    .str.contains(search_text, regex=False)
                    | review_frame["packet_id"].map(_format_display_value).str.lower().str.contains(search_text, regex=False)
                    | query_values.str.contains(search_text, regex=False)
                )
                review_frame = review_frame[review_mask]

            if review_frame.empty:
                st.warning("No rows matched the search text with current filters.")
            else:
                row_labels = (
                    review_frame.apply(
                        lambda row: (
                            f"{int(row['source_row_index'])} | {row['item_id']} | "
                            f"mode {row['target_error_mode']} | {row['model_config']} | {row.get('response_status', 'unknown')}"
                        ),
                        axis=1,
                    )
                    .tolist()
                )
                selected_row_label = st.selectbox("Select row", options=row_labels, index=0)
                selected_row = review_frame.loc[row_labels.index(selected_row_label)]

                detail_col_1, detail_col_2 = st.columns([1.1, 0.9])

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
                            {"field": "item_id", "value": selected_row.get("item_id")},
                            {"field": "packet_id", "value": selected_row.get("packet_id")},
                            {"field": "target_error_mode", "value": selected_row.get("target_error_mode")},
                            {"field": "response_status", "value": selected_row.get("response_status")},
                            {"field": "grade_passed", "value": selected_row.get("grade_passed")},
                            {
                                "field": "overall_required_graders_passed",
                                "value": selected_row.get("overall_required_graders_passed"),
                            },
                            {
                                "field": "overall_required_graders_reason",
                                "value": selected_row.get("overall_required_graders_reason"),
                            },
                            {"field": "latency_seconds", "value": selected_row.get("latency_seconds")},
                            {"field": "total_tokens", "value": selected_row.get("total_tokens")},
                        ]
                    )
                    st.dataframe(_prepare_display_frame(metadata_rows), width="stretch", hide_index=True, height=320)

                    selected_row_errors = pd.DataFrame()
                    if not errors_frame.empty:
                        selected_row_errors = errors_frame[
                            (errors_frame["model_config"] == selected_row.get("model_config"))
                            & (errors_frame["source_row_index"] == selected_row.get("source_row_index"))
                            & (errors_frame["item_id"] == selected_row.get("item_id"))
                        ].copy()

                    st.markdown("**Grader outcomes**")
                    grader_view_frame = _build_grader_view(
                        selected_row=selected_row,
                        grader_names=grader_names,
                    )
                    error_component_view = _build_error_component_view(selected_row_errors)
                    combined_grader_view = pd.concat([grader_view_frame, error_component_view], ignore_index=True)
                    st.dataframe(_prepare_display_frame(combined_grader_view), width="stretch", hide_index=True, height=250)

                st.markdown("**Error log entries for this row**")
                if not errors_frame.empty:
                    if selected_row_errors.empty:
                        st.caption("No error rows for this item.")
                    else:
                        st.dataframe(_prepare_display_frame(selected_row_errors), width="stretch", hide_index=True, height=220)
                else:
                    st.caption("No `errors.csv` file in this run.")

    with plots_tab:
        if filtered_predictions_frame.empty:
            st.info("No rows available for plots with current filters.")
        else:
            status_counts_frame = (
                filtered_predictions_frame["response_status"].fillna("missing").value_counts().rename_axis("status").reset_index(name="count")
            )
            status_figure = px.bar(
                status_counts_frame,
                x="status",
                y="count",
                color="status",
                title="Response status distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            status_figure.update_layout(xaxis_title="Response status", yaxis_title="Rows")
            st.plotly_chart(status_figure, width="stretch")

            failure_melt_frame = filtered_predictions_frame.melt(
                id_vars=["target_error_mode"],
                value_vars=["model_failed", "template_grade_failed", "required_graders_failed", "any_grader_failed"],
                var_name="failure_type",
                value_name="is_failure",
            )
            failure_melt_frame = failure_melt_frame[failure_melt_frame["is_failure"] == True]

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
                st.plotly_chart(failure_figure, width="stretch")

            grader_pass_rate_frame = _build_grader_pass_rate_frame(filtered_predictions_frame, grader_names)
            if not grader_pass_rate_frame.empty:
                grader_figure = px.bar(
                    grader_pass_rate_frame,
                    x="grader_name",
                    y="pass_rate_percent",
                    color="target_error_mode",
                    barmode="group",
                    hover_data=["evaluated_rows"],
                    title="Grader pass rate by error mode",
                    color_discrete_sequence=px.colors.qualitative.Pastel1,
                )
                grader_figure.update_layout(
                    xaxis_title="Grader",
                    yaxis_title="Pass rate (%)",
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(grader_figure, width="stretch")

            if "latency_seconds" in filtered_predictions_frame.columns:
                latency_frame = filtered_predictions_frame.dropna(subset=["latency_seconds"]).copy()
                if not latency_frame.empty:
                    latency_figure = px.box(
                        latency_frame,
                        x="target_error_mode",
                        y="latency_seconds",
                        color="target_error_mode",
                        points="outliers",
                        title="Latency distribution by mode",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                    )
                    latency_figure.update_layout(
                        xaxis_title="Target error mode",
                        yaxis_title="Latency (seconds)",
                        showlegend=False,
                    )
                    st.plotly_chart(latency_figure, width="stretch")

    with raw_tab:
        st.subheader("Raw exports")
        st.markdown("`metrics_summary.csv`")
        st.dataframe(_prepare_display_frame(metrics_frame), width="stretch", hide_index=True)
        st.markdown("`grader_metrics_summary.csv`")
        st.dataframe(_prepare_display_frame(grader_metrics_frame), width="stretch", hide_index=True)
        st.markdown("`errors.csv`")
        st.dataframe(_prepare_display_frame(errors_frame), width="stretch", hide_index=True)


def main() -> None:
    render_dashboard()


def launch() -> None:
    dashboard_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    from streamlit.web import cli as streamlit_cli

    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
