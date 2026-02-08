#!/usr/bin/env python3
"""Prepare structured evidence for synthetic-generation root-cause analysis."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

STATUS_COLUMN = "final_validation_status"
MODE_COLUMN = "target_error_mode"
REJECTION_REASON_COLUMN = "final_rejection_reason"
REJECTION_STAGE_COLUMN = "final_rejection_stage"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare synthetic-generation evidence tables and descriptive summaries. "
            "This script does not determine root causes or recommendations."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing data/, results/, and src/.",
    )
    parser.add_argument(
        "--run-timestamp",
        default="",
        help=(
            "Run timestamp prefix (for example 20260208T001140Z). "
            "If omitted, the newest complete run is used."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=6,
        help="Max examples to include per section in markdown output.",
    )
    return parser.parse_args()


def list_packet_ids(repo_root: Path) -> list[str]:
    packet_root = repo_root / "data" / "case_law_packets"
    packet_ids = sorted(
        packet_path.name
        for packet_path in packet_root.glob("packet_*")
        if packet_path.is_dir()
    )
    if not packet_ids:
        raise ValueError(f"No packet directories found under {packet_root}")
    return packet_ids


def list_run_prefixes_for_packet(results_root: Path, packet_id: str) -> set[str]:
    packet_results_dir = results_root / packet_id
    if not packet_results_dir.exists():
        return set()

    suffix = f"_{packet_id}"
    prefixes: set[str] = set()
    for run_dir in packet_results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if run_name.endswith(suffix):
            prefixes.add(run_name[: -len(suffix)])
    return prefixes


def resolve_run_timestamp(repo_root: Path, packet_ids: list[str], run_timestamp: str) -> str:
    results_root = repo_root / "results" / "synthetic_generation"

    if run_timestamp:
        for packet_id in packet_ids:
            run_dir = results_root / packet_id / f"{run_timestamp}_{packet_id}"
            if not run_dir.exists():
                raise ValueError(
                    f"Missing run directory for packet {packet_id}: {run_dir}"
                )
        return run_timestamp

    shared_prefixes: set[str] | None = None
    for packet_id in packet_ids:
        packet_prefixes = list_run_prefixes_for_packet(results_root, packet_id)
        if shared_prefixes is None:
            shared_prefixes = set(packet_prefixes)
        else:
            shared_prefixes = shared_prefixes.intersection(packet_prefixes)

    if not shared_prefixes:
        raise ValueError(
            "Could not find a complete run timestamp present across all packet directories."
        )

    return sorted(shared_prefixes)[-1]


def load_validation_table(
    repo_root: Path,
    packet_ids: list[str],
    run_timestamp: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    results_root = repo_root / "results" / "synthetic_generation"

    for packet_id in packet_ids:
        run_dir = results_root / packet_id / f"{run_timestamp}_{packet_id}"
        validation_csv = run_dir / "validation" / "validation_datapoints.csv"
        if not validation_csv.exists():
            raise ValueError(f"Missing validation file: {validation_csv}")

        validation_dataframe = pd.read_csv(validation_csv)
        validation_dataframe["packet"] = packet_id
        validation_dataframe["run_dir"] = str(run_dir)
        frames.append(validation_dataframe)

    return pd.concat(frames, ignore_index=True)


def normalize_status(validation_dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized_dataframe = validation_dataframe.copy()
    normalized_dataframe["status_normalized"] = (
        normalized_dataframe[STATUS_COLUMN].astype(str).str.lower()
    )
    return normalized_dataframe


def extract_message_prefix(trace_payload: dict) -> str:
    outputs = trace_payload.get("output")
    if not isinstance(outputs, list):
        return ""

    for output_item in outputs:
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") != "message":
            continue
        content_items = output_item.get("content")
        if not isinstance(content_items, list):
            continue

        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") != "output_text":
                continue
            text_value = content_item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value.strip().replace("\n", " ")[:220]

    return ""


def collect_trace_health(
    repo_root: Path,
    packet_ids: list[str],
    run_timestamp: str,
) -> pd.DataFrame:
    results_root = repo_root / "results" / "synthetic_generation"
    attempt_pattern = re.compile(r"attempt(\d+)")
    rows: list[dict] = []

    for packet_id in packet_ids:
        run_dir = results_root / packet_id / f"{run_timestamp}_{packet_id}"

        for stage_name in ["generation", "validation"]:
            trace_dir = run_dir / stage_name / "traces"
            if not trace_dir.exists():
                continue

            for trace_file in sorted(trace_dir.glob("*.json")):
                trace_text = trace_file.read_text(encoding="utf-8")
                try:
                    trace_payload = json.loads(trace_text)
                    json_parse_ok = True
                except json.JSONDecodeError:
                    rows.append(
                        {
                            "packet": packet_id,
                            "stage": stage_name,
                            "trace_file": str(trace_file),
                            "json_parse_ok": False,
                            "status": "json_decode_error",
                            "error": "json_decode_error",
                            "incomplete_reason": "",
                            "attempt": 1,
                            "model": "",
                            "input_tokens": None,
                            "output_tokens": None,
                            "reasoning_tokens": None,
                            "message_prefix": "",
                        }
                    )
                    continue

                usage = trace_payload.get("usage")
                if not isinstance(usage, dict):
                    usage = {}

                output_token_details = usage.get("output_tokens_details")
                if not isinstance(output_token_details, dict):
                    output_token_details = {}

                incomplete_reason = ""
                incomplete_details = trace_payload.get("incomplete_details")
                if isinstance(incomplete_details, dict):
                    reason_value = incomplete_details.get("reason")
                    if reason_value is not None:
                        incomplete_reason = str(reason_value)

                attempt_number = 1
                attempt_match = attempt_pattern.search(trace_file.name)
                if attempt_match:
                    attempt_number = int(attempt_match.group(1))

                rows.append(
                    {
                        "packet": packet_id,
                        "stage": stage_name,
                        "trace_file": str(trace_file),
                        "json_parse_ok": json_parse_ok,
                        "status": str(trace_payload.get("status", "")),
                        "error": "" if trace_payload.get("error") is None else str(trace_payload.get("error")),
                        "incomplete_reason": incomplete_reason,
                        "attempt": attempt_number,
                        "model": str(trace_payload.get("model", "")),
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "reasoning_tokens": output_token_details.get("reasoning_tokens"),
                        "message_prefix": extract_message_prefix(trace_payload),
                    }
                )

    return pd.DataFrame(rows)


def to_markdown_table(dataframe: pd.DataFrame, max_rows: int) -> str:
    if dataframe.empty:
        return "(none)"

    preview_dataframe = dataframe.head(max_rows).fillna("").copy()
    headers = list(preview_dataframe.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---" for _ in headers]) + " |",
    ]

    for _, row in preview_dataframe.iterrows():
        row_values = [str(row[column_name]).replace("\n", " ") for column_name in headers]
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def build_reasoning_evidence_tables(
    accepted_dataframe: pd.DataFrame,
    rejected_dataframe: pd.DataFrame,
    trace_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rejected_columns = [
        "item_id",
        "packet",
        MODE_COLUMN,
        "mode_name",
        REJECTION_STAGE_COLUMN,
        REJECTION_REASON_COLUMN,
        "deterministic_reason_codes",
        "llm_reason",
        "llm_risk_flags_json",
        "user_query",
        "scenario_facts_json",
        "expected_citation_groups_json",
        "deterministic_pass",
        "llm_verdict",
        "validation_request_status",
    ]
    rejected_evidence = rejected_dataframe.reindex(columns=rejected_columns).copy()

    accepted_columns = [
        "item_id",
        "packet",
        MODE_COLUMN,
        "mode_name",
        "llm_reason",
        "llm_risk_flags_json",
        "user_query",
        "scenario_facts_json",
        "expected_citation_groups_json",
        "deterministic_pass",
        "llm_verdict",
        "validation_request_status",
    ]
    accepted_evidence = accepted_dataframe.reindex(columns=accepted_columns).copy()

    trace_columns = [
        "packet",
        "stage",
        "status",
        "error",
        "incomplete_reason",
        "attempt",
        "model",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "trace_file",
        "message_prefix",
    ]
    trace_evidence = trace_dataframe.reindex(columns=trace_columns).copy()

    return rejected_evidence, accepted_evidence, trace_evidence


def write_analyst_workflow(output_dir: Path) -> None:
    workflow_text = """# Analyst Workflow (LLM/Manual Reasoning Required)

This folder provides parsed evidence only. It does **not** determine root causes.

## Important
- Do not treat `llm_risk_flags_json`, `llm_reason`, or `final_rejection_reason` as automatic ground truth.
- Use those fields as clues, then verify by reading `user_query`, `scenario_facts_json`, and `expected_citation_groups_json`.
- Derive your own error modes and root causes from evidence.

## Recommended process
1. Start with `rejected_reasoning_evidence.csv`.
2. Group by mode (`target_error_mode`) and read at least several examples per mode.
3. Identify repeated failure patterns in your own words.
4. Cross-check with `accepted_reasoning_evidence.csv` to avoid overfitting to rejected-only patterns.
5. Validate operational health with `trace_reasoning_evidence.csv`.
6. Produce prompt-only recommendations and clearly mark them as not implemented.
"""
    (output_dir / "analyst_workflow.md").write_text(workflow_text, encoding="utf-8")


def write_report(
    output_dir: Path,
    run_timestamp: str,
    packet_ids: list[str],
    validation_dataframe: pd.DataFrame,
    accepted_dataframe: pd.DataFrame,
    rejected_dataframe: pd.DataFrame,
    trace_dataframe: pd.DataFrame,
    rejected_evidence: pd.DataFrame,
    accepted_evidence: pd.DataFrame,
    trace_evidence: pd.DataFrame,
    max_examples: int,
) -> None:
    # Remove legacy recommendation artifact so stale heuristic outputs are not
    # mistaken for model-derived conclusions in reused run folders.
    legacy_recommendation_path = output_dir / "prompt_recommendations.csv"
    if legacy_recommendation_path.exists():
        legacy_recommendation_path.unlink()

    overall_counts = (
        validation_dataframe["status_normalized"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )

    by_packet_status = (
        validation_dataframe.groupby(["packet", "status_normalized"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    by_mode_status = (
        validation_dataframe.groupby([MODE_COLUMN, "status_normalized"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    rejected_by_mode = (
        rejected_dataframe[MODE_COLUMN]
        .value_counts()
        .rename_axis(MODE_COLUMN)
        .reset_index(name="count")
    )

    rejected_by_stage = (
        rejected_dataframe[REJECTION_STAGE_COLUMN]
        .fillna("<missing>")
        .value_counts()
        .rename_axis(REJECTION_STAGE_COLUMN)
        .reset_index(name="count")
    )

    trace_status = (
        trace_dataframe.groupby(["stage", "status"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    query_length_series = accepted_dataframe["user_query"].astype(str).str.len()
    query_length_stats = query_length_series.describe(percentiles=[0.5, 0.9, 0.95])

    summary_metrics_rows = [
        {"metric": "run_timestamp", "value": run_timestamp},
        {"metric": "packet_count", "value": len(packet_ids)},
        {"metric": "total_items", "value": int(validation_dataframe.shape[0])},
        {"metric": "accepted_items", "value": int(accepted_dataframe.shape[0])},
        {"metric": "rejected_items", "value": int(rejected_dataframe.shape[0])},
        {"metric": "trace_file_count", "value": int(trace_dataframe.shape[0])},
        {
            "metric": "trace_completed_count",
            "value": int((trace_dataframe["status"] == "completed").sum()),
        },
        {
            "metric": "trace_retry_file_count",
            "value": int((trace_dataframe["attempt"] > 1).sum()),
        },
    ]
    summary_metrics_dataframe = pd.DataFrame(summary_metrics_rows)

    summary_metrics_dataframe.to_csv(output_dir / "summary_metrics.csv", index=False)
    validation_dataframe.to_csv(output_dir / "validation_all.csv", index=False)
    accepted_dataframe.to_csv(output_dir / "accepted_items.csv", index=False)
    rejected_dataframe.to_csv(output_dir / "rejected_items.csv", index=False)
    trace_dataframe.to_csv(output_dir / "trace_health.csv", index=False)
    rejected_evidence.to_csv(output_dir / "rejected_reasoning_evidence.csv", index=False)
    accepted_evidence.to_csv(output_dir / "accepted_reasoning_evidence.csv", index=False)
    trace_evidence.to_csv(output_dir / "trace_reasoning_evidence.csv", index=False)

    lines: list[str] = []
    lines.append("# Synthetic Generation Evidence Report")
    lines.append("")
    lines.append(f"- run_timestamp: `{run_timestamp}`")
    lines.append(f"- packet_ids: `{', '.join(packet_ids)}`")
    lines.append("")

    lines.append("## Important")
    lines.append(
        "This report is descriptive only. Root-cause diagnosis and recommendations must be derived by reading evidence tables, not by trusting pre-labeled reason fields."
    )
    lines.append("")

    lines.append("## Overall Outcome")
    lines.append(to_markdown_table(overall_counts, max_rows=max_examples))
    lines.append("")

    lines.append("## Outcome by Packet")
    lines.append(to_markdown_table(by_packet_status, max_rows=50))
    lines.append("")

    lines.append("## Outcome by Error Mode")
    lines.append(to_markdown_table(by_mode_status, max_rows=50))
    lines.append("")

    lines.append("## Rejected Distribution")
    lines.append(to_markdown_table(rejected_by_mode, max_rows=50))
    lines.append("")

    lines.append("### Rejected by Stage")
    lines.append(to_markdown_table(rejected_by_stage, max_rows=50))
    lines.append("")

    lines.append("## Accepted Query Length Stats")
    lines.append(
        to_markdown_table(
            query_length_stats.rename_axis("stat").reset_index(name="value"),
            max_rows=20,
        )
    )
    lines.append("")

    lines.append("## Trace Health")
    lines.append(to_markdown_table(trace_status, max_rows=50))
    lines.append("")

    lines.append("## Evidence Tables")
    lines.append("Use these tables for LLM/manual reasoning:")
    lines.append("- `rejected_reasoning_evidence.csv`")
    lines.append("- `accepted_reasoning_evidence.csv`")
    lines.append("- `trace_reasoning_evidence.csv`")
    lines.append("- `analyst_workflow.md`")
    lines.append("")

    lines.append("### Rejected Evidence Examples")
    lines.append(to_markdown_table(rejected_evidence, max_rows=max_examples))
    lines.append("")

    lines.append("### Accepted Evidence Examples")
    lines.append(to_markdown_table(accepted_evidence, max_rows=max_examples))
    lines.append("")

    lines.append("### Trace Evidence Examples")
    lines.append(to_markdown_table(trace_evidence, max_rows=max_examples))
    lines.append("")

    (output_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    repo_root = Path(arguments.repo_root).resolve()

    packet_ids = list_packet_ids(repo_root)
    run_timestamp = resolve_run_timestamp(repo_root, packet_ids, arguments.run_timestamp)

    validation_dataframe = load_validation_table(repo_root, packet_ids, run_timestamp)
    validation_dataframe = normalize_status(validation_dataframe)

    accepted_dataframe = validation_dataframe[
        validation_dataframe["status_normalized"] == "accepted"
    ].copy()
    rejected_dataframe = validation_dataframe[
        validation_dataframe["status_normalized"] == "rejected"
    ].copy()

    trace_dataframe = collect_trace_health(repo_root, packet_ids, run_timestamp)

    rejected_evidence, accepted_evidence, trace_evidence = build_reasoning_evidence_tables(
        accepted_dataframe,
        rejected_dataframe,
        trace_dataframe,
    )

    output_dir = repo_root / "results" / "synthetic_generation_audit" / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    write_analyst_workflow(output_dir)
    write_report(
        output_dir=output_dir,
        run_timestamp=run_timestamp,
        packet_ids=packet_ids,
        validation_dataframe=validation_dataframe,
        accepted_dataframe=accepted_dataframe,
        rejected_dataframe=rejected_dataframe,
        trace_dataframe=trace_dataframe,
        rejected_evidence=rejected_evidence,
        accepted_evidence=accepted_evidence,
        trace_evidence=trace_evidence,
        max_examples=arguments.max_examples,
    )

    print(f"run_timestamp={run_timestamp}")
    print(f"report={output_dir / 'analysis_summary.md'}")
    print(f"rejected_evidence={output_dir / 'rejected_reasoning_evidence.csv'}")
    print(f"accepted_evidence={output_dir / 'accepted_reasoning_evidence.csv'}")
    print(f"trace_evidence={output_dir / 'trace_reasoning_evidence.csv'}")


if __name__ == "__main__":
    main()
