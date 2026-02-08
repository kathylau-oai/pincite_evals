import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from pincite_evals.eval_runner import (
    DEFAULT_DRAFTING_SYSTEM_PROMPT,
    ModelConfig,
    _build_drafting_user_prompt,
    _build_grader_context,
    _build_predictions_and_grades_debug_export,
    _build_predictions_with_grades_export,
    _build_response_request,
    _parse_args,
    _compute_distribution_stats,
    _estimate_inter_token_latency_seconds,
    _evaluate_single_row,
    _prepare_spreadsheet_friendly_export_frame,
    _prepare_dataset,
    _select_graders_for_mode,
    _template_grade,
    _write_spreadsheet_friendly_csv,
)


def test_build_response_request_includes_temperature_when_reasoning_none():
    model_config = ModelConfig(
        name="baseline",
        model="gpt-5.2",
        reasoning_effort="none",
        temperature=0.2,
        system_prompt="You are a test assistant.",
    )

    request = _build_response_request(model_config, "Test prompt")

    assert request["model"] == "gpt-5.2"
    assert request["service_tier"] == "priority"
    assert request["reasoning"] == {"effort": "none"}
    assert request["temperature"] == 0.2


def test_build_response_request_omits_temperature_when_reasoning_enabled():
    model_config = ModelConfig(
        name="reasoning_config",
        model="gpt-5.2",
        reasoning_effort="high",
        temperature=0.7,
        system_prompt="You are a test assistant.",
    )

    request = _build_response_request(model_config, "Test prompt")

    assert request["service_tier"] == "priority"
    assert request["reasoning"] == {"effort": "high"}
    assert "temperature" not in request


def test_parse_args_defaults_output_root_and_artifact_level(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pincite-eval"])
    args = _parse_args()

    assert args.output_root == "results/experiments"
    assert args.artifact_level == "standard"


def test_template_grade_exact_match_and_not_graded_cases():
    pass_grade = _template_grade("Final answer", "Final answer")
    assert pass_grade["grade_label"] == "pass"
    assert pass_grade["grade_passed"] is True
    assert pass_grade["grade_score"] == 1.0

    fail_grade = _template_grade("Different answer", "Expected answer")
    assert fail_grade["grade_label"] == "fail"
    assert fail_grade["grade_passed"] is False
    assert fail_grade["grade_score"] == 0.0

    skipped_grade = _template_grade("Any answer", None)
    assert skipped_grade["grade_label"] == "not_graded"
    assert skipped_grade["grade_passed"] is None
    assert skipped_grade["grade_score"] is None


def test_distribution_stats_computes_expected_values():
    values = pd.Series([1, 2, 3, 4, 5])
    stats = _compute_distribution_stats(values, "latency")

    assert stats["latency_count"] == 5
    assert stats["latency_avg"] == 3.0
    assert stats["latency_p50"] == 3.0
    assert stats["latency_p90"] == 4.6
    assert stats["latency_p95"] == 4.8
    assert stats["latency_p99"] == 4.96


def test_estimate_inter_token_latency_seconds_uses_e2e_over_output_tokens():
    inter_token_latency = _estimate_inter_token_latency_seconds(
        latency_seconds=8.0,
        output_tokens=4.0,
    )

    assert inter_token_latency == 2.0


def test_estimate_inter_token_latency_seconds_returns_none_without_valid_tokens():
    assert _estimate_inter_token_latency_seconds(latency_seconds=5.0, output_tokens=0.0) is None
    assert _estimate_inter_token_latency_seconds(latency_seconds=5.0, output_tokens=None) is None
    assert _estimate_inter_token_latency_seconds(latency_seconds=None, output_tokens=5.0) is None


def test_evaluate_single_row_estimates_inter_token_latency_from_e2e_and_output_tokens(tmp_path):
    row = pd.Series(
        {
            "item_id": "row_0",
            "user_query": "Draft a memo.",
            "expected_output": "Memo output",
            "packet_corpus_text": '<BLOCK id="DOC001.P001.B01">Authority text.</BLOCK>',
            "scenario_facts_parsed": ["Fact one"],
        }
    )

    model_config = ModelConfig(
        name="baseline",
        model="gpt-5.2",
        reasoning_effort="none",
        temperature=0.0,
        system_prompt="You are a test assistant.",
    )

    def fake_call_model(_request):
        return {
            "response_id": "resp_123",
            "status": "completed",
            "incomplete_reason": None,
            "output_text": "Memo output",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "latency_seconds": 8.0,
            "ttft_seconds": 0.7,
            "raw_response": {"id": "resp_123"},
        }

    result_row, inter_token_latencies = _evaluate_single_row(
        row=row,
        source_row_index=0,
        model_config=model_config,
        call_model=fake_call_model,
        raw_response_dir=tmp_path,
        prompt_column="user_query",
        id_column="item_id",
        expected_output_column="expected_output",
        dry_run=False,
    )

    assert inter_token_latencies == [2.0]
    assert result_row["inter_token_latency_avg_seconds"] == 2.0
    assert result_row["inter_token_event_count"] == 4


def test_prepare_dataset_creates_id_and_expected_output_columns(tmp_path):
    input_csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"user_query": ["p1", "p2"]}).to_csv(input_csv_path, index=False)

    args = Namespace(
        input_csv=str(input_csv_path),
        id_column="item_id",
        expected_output_column="expected_output",
        max_samples=None,
    )

    dataset = _prepare_dataset(args)

    assert "item_id" in dataset.columns
    assert "expected_output" in dataset.columns
    assert "user_query" in dataset.columns
    assert dataset["item_id"].tolist() == ["row_0", "row_1"]


def test_prepare_dataset_requires_user_query_column(tmp_path):
    input_csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"prompt": ["p1"]}).to_csv(input_csv_path, index=False)

    args = Namespace(
        input_csv=str(input_csv_path),
        id_column="item_id",
        expected_output_column="expected_output",
        max_samples=None,
    )

    with pytest.raises(ValueError, match="User query column 'user_query'"):
        _prepare_dataset(args)


def test_build_grader_context_parses_expected_groups_from_prediction_json():
    row = {
        "item_id": "packet_1_A_01",
        "packet_id": "packet_1",
        "target_error_mode": "A",
        "query_id": "q_fak_0001",
        "as_of_date": "2026-02-07",
        "model_output": "Citation (DOC001.P001.B01).",
        "scenario_facts_json": "[\"fact one\"]",
        "grading_contract_json": "{\"expected_citation_groups\": [[\"DOC001.P001.B01\"]]}",
        "expected_citation_groups_json": "[[\"DOC001.P001.B01\"]]",
    }

    context = _build_grader_context(row, block_text_by_packet={"packet_1": {"DOC001.P001.B01": "Block text"}})

    assert context["expected_citation_groups"] == [["DOC001.P001.B01"]]
    assert context["test_case_context"]["scenario_facts"] == ["fact one"]
    assert context["allow_unexpected_citations_when_no_expected_groups"] is False


def test_build_grader_context_allows_mode_a_unexpected_when_expected_groups_empty():
    row = {
        "item_id": "packet_1_A_01",
        "packet_id": "packet_1",
        "target_error_mode": "A",
        "query_id": "q_fak_0001",
        "as_of_date": "2026-02-07",
        "model_output": "Citation (DOC001.P001.B01).",
        "scenario_facts_json": "[]",
        "grading_contract_json": "{\"expected_citation_groups\": []}",
        "expected_citation_groups_json": "[]",
    }

    context = _build_grader_context(row, block_text_by_packet={"packet_1": {"DOC001.P001.B01": "Block text"}})

    assert context["expected_citation_groups"] == []
    assert context["allow_unexpected_citations_when_no_expected_groups"] is True


def test_select_graders_for_mode_excludes_expected_citation_presence():
    assert _select_graders_for_mode("A") == ["citation_fidelity_llm_judge"]
    assert _select_graders_for_mode("C") == ["citation_overextension_llm_judge"]
    assert _select_graders_for_mode("D") == ["precedence_llm_judge"]
    assert _select_graders_for_mode("B") == []


def test_build_predictions_with_grades_export_includes_per_grader_columns_and_reasons():
    predictions_with_grades_frame = pd.DataFrame(
        [
            {
                "model_config": "baseline",
                "source_row_index": 0,
                "item_id": "item_0",
                "packet_id": "packet_1",
                "query_id": "q_0",
                "as_of_date": "2026-02-08",
                "target_error_mode": "A",
                "source_user_query": "Query 0",
                "model_output": "Output 0",
                "response_status": "completed",
                "rendered_user_prompt": "very large prompt payload",
                "overall_required_graders_passed": True,
            },
            {
                "model_config": "baseline",
                "source_row_index": 1,
                "item_id": "item_1",
                "packet_id": "packet_1",
                "query_id": "q_1",
                "as_of_date": "2026-02-08",
                "target_error_mode": "C",
                "source_user_query": "Query 1",
                "model_output": "Output 1",
                "response_status": "completed",
                "rendered_user_prompt": "another large prompt payload",
                "overall_required_graders_passed": False,
            },
        ]
    )

    grader_frame = pd.DataFrame(
        [
            {
                "model_config": "baseline",
                "source_row_index": 0,
                "item_id": "item_0",
                "grader_name": "expected_citation_presence",
                "grader_status": "completed",
                "grader_passed": True,
                "grader_error": None,
                "grader_details_json": "{}",
            },
            {
                "model_config": "baseline",
                "source_row_index": 1,
                "item_id": "item_1",
                "grader_name": "expected_citation_presence",
                "grader_status": "completed",
                "grader_passed": False,
                "grader_error": None,
                "grader_details_json": "{\"missing_groups\": [[\"DOC001.P001.B01\"]], \"unexpected_predictions\": []}",
            },
            {
                "model_config": "baseline",
                "source_row_index": 1,
                "item_id": "item_1",
                "grader_name": "citation_overextension_llm_judge",
                "grader_status": "completed",
                "grader_passed": False,
                "grader_error": None,
                "grader_details_json": "{\"reason\": \"Claim overstates the authority.\"}",
            },
        ]
    )

    export_frame = _build_predictions_with_grades_export(predictions_with_grades_frame, grader_frame)

    expected_base_columns = {
        "model_config",
        "source_row_index",
        "item_id",
        "packet_id",
        "query_id",
        "as_of_date",
        "target_error_mode",
        "user_query",
        "model_output",
        "response_status",
        "overall_required_graders_passed",
        "overall_required_graders_reason",
    }
    assert expected_base_columns.issubset(set(export_frame.columns))
    assert "rendered_user_prompt" not in export_frame.columns
    assert export_frame.loc[0, "overall_required_graders_reason"] == "All required graders passed."
    assert "expected_citation_presence: Missing groups: 1; unexpected citations: 0." in export_frame.loc[
        1, "overall_required_graders_reason"
    ]
    assert "citation_overextension_llm_judge: Claim overstates the authority." in export_frame.loc[
        1, "overall_required_graders_reason"
    ]

    # Per-grader granular columns are required for easier manual audit.
    assert export_frame.loc[0, "grader_expected_citation_presence_status"] == "completed"
    assert bool(export_frame.loc[0, "grader_expected_citation_presence_passed"]) is True
    assert export_frame.loc[0, "grader_expected_citation_presence_reason"] == "Grader did not provide a reason."

    assert export_frame.loc[1, "grader_expected_citation_presence_status"] == "completed"
    assert bool(export_frame.loc[1, "grader_expected_citation_presence_passed"]) is False
    assert export_frame.loc[1, "grader_expected_citation_presence_reason"] == "Missing groups: 1; unexpected citations: 0."

    assert export_frame.loc[1, "grader_citation_overextension_llm_judge_status"] == "completed"
    assert bool(export_frame.loc[1, "grader_citation_overextension_llm_judge_passed"]) is False
    assert export_frame.loc[1, "grader_citation_overextension_llm_judge_reason"] == "Claim overstates the authority."


def test_build_predictions_and_grades_debug_export_keeps_only_granular_grader_pass_fail_and_reason():
    predictions_frame = pd.DataFrame(
        [
            {
                "model_config": "baseline",
                "source_row_index": 0,
                "item_id": "item_0",
                "packet_id": "packet_1",
                "query_id": "q_0",
                "as_of_date": "2026-02-08",
                "target_error_mode": "A",
                "source_user_query": "Query 0",
                "model_output": "Output 0",
                "response_status": "completed",
                "overall_required_graders_passed": True,
            },
            {
                "model_config": "baseline",
                "source_row_index": 1,
                "item_id": "item_1",
                "packet_id": "packet_1",
                "query_id": "q_1",
                "as_of_date": "2026-02-08",
                "target_error_mode": "C",
                "source_user_query": "Query 1",
                "model_output": "Output 1",
                "response_status": "completed",
                "overall_required_graders_passed": False,
            },
        ]
    )

    grader_frame = pd.DataFrame(
        [
            {
                "model_config": "baseline",
                "source_row_index": 0,
                "item_id": "item_0",
                "grader_name": "expected_citation_presence",
                "grader_status": "completed",
                "grader_passed": True,
                "grader_details_json": "{}",
            },
            {
                "model_config": "baseline",
                "source_row_index": 1,
                "item_id": "item_1",
                "grader_name": "citation_overextension_llm_judge",
                "grader_status": "completed",
                "grader_passed": False,
                "grader_details_json": "{\"reason\": \"Claim overstates the authority.\"}",
            },
        ]
    )

    export_frame = _build_predictions_and_grades_debug_export(predictions_frame, grader_frame)

    assert "user_query" in export_frame.columns
    assert "source_user_query" not in export_frame.columns
    assert "overall_required_graders_passed" not in export_frame.columns
    assert "overall_required_graders_reason" not in export_frame.columns

    assert bool(export_frame.loc[0, "grader_expected_citation_presence_passed"]) is True
    assert export_frame.loc[0, "grader_expected_citation_presence_reason"] == "Grader did not provide a reason."
    assert bool(export_frame.loc[1, "grader_citation_overextension_llm_judge_passed"]) is False
    assert export_frame.loc[1, "grader_citation_overextension_llm_judge_reason"] == "Claim overstates the authority."

    unexpected_grader_columns = [
        column_name
        for column_name in export_frame.columns
        if column_name.startswith("grader_")
        and (
            column_name.endswith("_status")
            or column_name.endswith("_score")
            or column_name.endswith("_label")
            or column_name.endswith("_details_json")
        )
    ]
    assert unexpected_grader_columns == []


def test_prepare_spreadsheet_friendly_export_frame_normalizes_embedded_newlines():
    export_frame = pd.DataFrame(
        [
            {
                "item_id": "item_0",
                "model_output": "Line one.\nLine two.\r\nLine three.\rLine four.",
                "score": 1.0,
            }
        ]
    )

    spreadsheet_frame = _prepare_spreadsheet_friendly_export_frame(export_frame)

    assert spreadsheet_frame.loc[0, "model_output"] == "Line one.\\nLine two.\\nLine three.\\nLine four."
    assert spreadsheet_frame.loc[0, "score"] == 1.0


def test_write_spreadsheet_friendly_csv_writes_bom_and_round_trippable_rows(tmp_path):
    output_csv_path = tmp_path / "predictions_with_grades.csv"
    export_frame = pd.DataFrame(
        [
            {
                "item_id": "item_0",
                "model_output": "First line.\nSecond line.",
                "overall_required_graders_reason": "all good",
            }
        ]
    )

    _write_spreadsheet_friendly_csv(export_frame=export_frame, output_csv_path=output_csv_path)

    raw_bytes = output_csv_path.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")

    loaded_frame = pd.read_csv(output_csv_path, encoding="utf-8-sig")
    assert loaded_frame.loc[0, "model_output"] == "First line.\\nSecond line."
    assert loaded_frame.shape == (1, 3)


def test_build_drafting_user_prompt_renders_user_template_variables():
    user_prompt_template = (
        "Task:\n{{ source_user_query }}\n\n"
        "Facts:\n{{ scenario_facts_block }}\n\n"
        "Corpus:\n{{ packet_corpus_text }}"
    )

    rendered_prompt = _build_drafting_user_prompt(
        source_user_query="Write memo section",
        scenario_facts=["Fact one", "Fact two"],
        packet_corpus_text='<DOCUMENT id=\"DOC001\">...</DOCUMENT>',
        user_prompt_template=user_prompt_template,
    )

    assert "Task:\nWrite memo section" in rendered_prompt
    assert "Facts:\n- Fact one\n- Fact two" in rendered_prompt
    assert 'Corpus:\n<DOCUMENT id="DOC001">...</DOCUMENT>' in rendered_prompt


def test_default_system_prompt_contains_memo_headings_and_footnote_requirements():
    assert "# Question Presented" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "# Brief Answer" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "# Rule / Governing Standard" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "# Analysis" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "# Counterarguments / Distinguishing" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "# Recommendation + Risk/Uncertainty" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "# Citations (verbatim list)" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "Footnote requirements" in DEFAULT_DRAFTING_SYSTEM_PROMPT
    assert "DOC###.P###.B##" in DEFAULT_DRAFTING_SYSTEM_PROMPT
