from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from pincite_evals.eval_runner import (
    ModelConfig,
    _build_grader_context,
    _build_response_request,
    _compute_distribution_stats,
    _estimate_inter_token_latency_seconds,
    _evaluate_single_row,
    _prepare_dataset,
    _template_grade,
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
