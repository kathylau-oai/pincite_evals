from pathlib import Path

import pandas as pd

from pincite_evals.experiment_review_dashboard import (
    _build_grader_view,
    _first_available_column,
    _prepare_display_frame,
    add_failure_columns,
    apply_filters,
    list_experiment_runs,
    merge_error_summary,
)


def test_list_experiment_runs_prefers_latest_timestamp(tmp_path: Path):
    older = tmp_path / "gpt41_none_full_20260207_235959"
    newer = tmp_path / "gpt51_none_full_20260208_004954"
    fallback = tmp_path / "manual_debug_run"

    (older / "final").mkdir(parents=True)
    (newer / "final").mkdir(parents=True)
    fallback.mkdir()
    (older / "final" / "predictions_with_grades.csv").write_text("item_id\nrow_1\n", encoding="utf-8")
    (newer / "final" / "predictions_with_grades.csv").write_text("item_id\nrow_1\n", encoding="utf-8")

    ordered_runs = list_experiment_runs(tmp_path)

    assert ordered_runs[0].name == newer.name
    assert older.name in [run.name for run in ordered_runs]
    assert fallback.name not in [run.name for run in ordered_runs]


def test_add_failure_columns_marks_model_grade_and_grader_failures():
    predictions_frame = pd.DataFrame(
        [
            {
                "model_config": "default",
                "target_error_mode": "A",
                "response_status": "completed",
                "grade_passed": True,
                "overall_required_graders_passed": True,
                "grader_citation_fidelity_llm_judge_passed": True,
            },
            {
                "model_config": "default",
                "target_error_mode": "C",
                "response_status": "error",
                "grade_passed": False,
                "overall_required_graders_passed": False,
                "grader_citation_fidelity_llm_judge_passed": False,
            },
        ]
    )

    enriched_frame = add_failure_columns(predictions_frame, ["citation_fidelity_llm_judge"])

    assert bool(enriched_frame.loc[0, "has_any_failure"]) is False
    assert bool(enriched_frame.loc[1, "model_failed"]) is True
    assert bool(enriched_frame.loc[1, "template_grade_failed"]) is True
    assert bool(enriched_frame.loc[1, "required_graders_failed"]) is True
    assert bool(enriched_frame.loc[1, "any_grader_failed"]) is True


def test_merge_error_summary_aggregates_error_rows():
    predictions_frame = pd.DataFrame(
        [
            {"model_config": "default", "source_row_index": 0, "item_id": "item_0"},
            {"model_config": "default", "source_row_index": 1, "item_id": "item_1"},
        ]
    )
    errors_frame = pd.DataFrame(
        [
            {
                "model_config": "default",
                "source_row_index": 1,
                "item_id": "item_1",
                "component_name": "citation_fidelity_llm_judge",
                "status": "error",
            },
            {
                "model_config": "default",
                "source_row_index": 1,
                "item_id": "item_1",
                "component_name": "precedence_llm_judge",
                "status": "skipped_model_error",
            },
        ]
    )

    merged_frame = merge_error_summary(predictions_frame, errors_frame)

    assert bool(merged_frame.loc[0, "has_error_row"]) is False
    assert merged_frame.loc[1, "error_row_count"] == 2
    assert bool(merged_frame.loc[1, "has_error_row"]) is True
    assert "citation_fidelity_llm_judge" in merged_frame.loc[1, "error_components"]


def test_apply_filters_respects_failure_and_mode_filters():
    predictions_frame = pd.DataFrame(
        [
            {
                "model_config": "default",
                "target_error_mode": "A",
                "source_row_index": 0,
                "item_id": "item_0",
                "has_any_failure": False,
                "model_failed": False,
                "template_grade_failed": False,
                "required_graders_failed": False,
                "any_grader_failed": False,
            },
            {
                "model_config": "default",
                "target_error_mode": "C",
                "source_row_index": 1,
                "item_id": "item_1",
                "has_any_failure": True,
                "model_failed": True,
                "template_grade_failed": True,
                "required_graders_failed": False,
                "any_grader_failed": False,
            },
            {
                "model_config": "baseline",
                "target_error_mode": "A",
                "source_row_index": 2,
                "item_id": "item_2",
                "has_any_failure": True,
                "model_failed": False,
                "template_grade_failed": False,
                "required_graders_failed": True,
                "any_grader_failed": True,
            },
        ]
    )

    filtered_frame = apply_filters(
        predictions_frame=predictions_frame,
        selected_model_configs=["default"],
        selected_error_modes=["C"],
        only_failures=True,
        selected_failure_columns=["model_failed", "template_grade_failed"],
    )

    assert filtered_frame.shape[0] == 1
    assert filtered_frame.iloc[0]["item_id"] == "item_1"


def test_first_available_column_prefers_first_present_name():
    frame = pd.DataFrame({"user_query": ["hello"], "model_output": ["world"]})
    column_name = _first_available_column(frame, ["source_user_query", "user_query"])
    assert column_name == "user_query"


def test_build_grader_view_falls_back_to_required_summary_when_grader_columns_missing():
    selected_row = pd.Series(
        {
            "overall_required_graders_passed": False,
            "overall_required_graders_reason": "expected_citation_presence: Missing required citation.",
        }
    )

    grader_frame = _build_grader_view(selected_row=selected_row, grader_names=[])

    assert grader_frame.shape[0] == 1
    assert grader_frame.iloc[0]["grader"] == "required_graders_summary"
    assert grader_frame.iloc[0]["status"] == "FAIL"
    assert "Missing required citation" in grader_frame.iloc[0]["detail"]


def test_prepare_display_frame_stringifies_mixed_object_values():
    frame = pd.DataFrame({"value": ["text", True, 3, None]})
    display_frame = _prepare_display_frame(frame)
    assert display_frame["value"].tolist() == ["text", "True", "3", ""]
