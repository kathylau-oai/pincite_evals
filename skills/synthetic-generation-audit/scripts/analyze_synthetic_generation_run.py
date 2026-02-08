#!/usr/bin/env python3
"""Audit synthetic generation outputs across packets and emit a reproducible report."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

STATUS_COLUMN = "final_validation_status"
MODE_COLUMN = "target_error_mode"
REJECTION_REASON_COLUMN = "final_rejection_reason"
REJECTION_STAGE_COLUMN = "final_rejection_stage"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze synthetic generation run outputs and write a markdown report with CSV artifacts."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing data/, results/, and src/.",
    )
    parser.add_argument(
        "--run-timestamp",
        default="",
        help="Run timestamp prefix (for example 20260208T001140Z). If omitted, the newest complete run is used.",
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
        if not run_name.endswith(suffix):
            continue
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
    repo_root: Path, packet_ids: list[str], run_timestamp: str
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


def parse_json_list(text_value: object) -> list[str]:
    if text_value is None:
        return []

    text = str(text_value).strip()
    if not text or text.lower() == "nan":
        return []

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        return []

    return [str(item) for item in parsed]


def build_rejection_breakdowns(
    rejected_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str], Counter[str]]:
    by_mode = (
        rejected_dataframe[MODE_COLUMN]
        .value_counts()
        .rename_axis("target_error_mode")
        .reset_index(name="count")
    )
    by_stage = (
        rejected_dataframe[REJECTION_STAGE_COLUMN]
        .fillna("<missing>")
        .value_counts()
        .rename_axis("rejection_stage")
        .reset_index(name="count")
    )

    risk_counter: Counter[str] = Counter()
    for risk_flags_text in rejected_dataframe["llm_risk_flags_json"].dropna():
        for risk_flag in parse_json_list(risk_flags_text):
            risk_counter[risk_flag] += 1

    deterministic_counter: Counter[str] = Counter()
    for reason_codes_text in rejected_dataframe["deterministic_reason_codes"].dropna():
        reason_codes = str(reason_codes_text).strip()
        if not reason_codes:
            continue
        for reason_code in reason_codes.split("|"):
            cleaned_reason_code = reason_code.strip()
            if cleaned_reason_code:
                deterministic_counter[cleaned_reason_code] += 1

    return by_mode, by_stage, risk_counter, deterministic_counter


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
            if not isinstance(text_value, str):
                continue
            stripped_text = text_value.strip()
            if stripped_text:
                return stripped_text.replace("\n", " ")[:220]

    return ""


def collect_trace_health(
    repo_root: Path, packet_ids: list[str], run_timestamp: str
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

                incomplete_details = trace_payload.get("incomplete_details")
                incomplete_reason = ""
                if isinstance(incomplete_details, dict):
                    incomplete_reason_value = incomplete_details.get("reason")
                    if incomplete_reason_value is not None:
                        incomplete_reason = str(incomplete_reason_value)

                attempt_match = attempt_pattern.search(trace_file.name)
                attempt_number = 1
                if attempt_match:
                    attempt_number = int(attempt_match.group(1))

                rows.append(
                    {
                        "packet": packet_id,
                        "stage": stage_name,
                        "trace_file": str(trace_file),
                        "json_parse_ok": True,
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

    preview_dataframe = dataframe.head(max_rows).copy()
    preview_dataframe = preview_dataframe.fillna("")

    headers = list(preview_dataframe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---" for _ in headers]) + " |",
    ]

    for _, row in preview_dataframe.iterrows():
        values = [str(row[column_name]).replace("\n", " ") for column_name in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_prompt_recommendations(
    rejected_dataframe: pd.DataFrame,
    risk_counter: Counter[str],
    deterministic_counter: Counter[str],
) -> pd.DataFrame:
    recommendations: list[dict] = []

    if risk_counter["ambiguous_grading_contract"] > 0 or risk_counter["internal_inconsistency"] > 0:
        recommendations.append(
            {
                "priority": "P1",
                "signal": "ambiguous_grading_contract/internal_inconsistency",
                "evidence_count": int(
                    risk_counter["ambiguous_grading_contract"]
                    + risk_counter["internal_inconsistency"]
                ),
                "prompt_recommendation": "Add a hard self-consistency audit checklist to generation system prompts requiring trigger-note, caution, and expected-citation coherence.",
                "scope": "generation system prompts (A/C/D)",
                "implementation_status": "recommendation_only_do_not_implement",
            }
        )

    mode_a_rejected = rejected_dataframe[rejected_dataframe[MODE_COLUMN] == "A"].copy()
    if not mode_a_rejected.empty:
        recommendations.append(
            {
                "priority": "P1",
                "signal": "mode_A_absence_truthfulness",
                "evidence_count": int(mode_a_rejected.shape[0]),
                "prompt_recommendation": "In Mode A prompt, require absence truthfulness: if packet has partial discussion of requested authority, trigger note must label it partial/limited support, not absent.",
                "scope": "fake_citations/system.txt",
                "implementation_status": "recommendation_only_do_not_implement",
            }
        )

    mode_c_rejected = rejected_dataframe[rejected_dataframe[MODE_COLUMN] == "C"].copy()
    if not mode_c_rejected.empty:
        recommendations.append(
            {
                "priority": "P1",
                "signal": "mode_C_scope_anchor_gap",
                "evidence_count": int(mode_c_rejected.shape[0]),
                "prompt_recommendation": "Require expected_citation_groups to include one base-rule anchor and one limiter/exception anchor, and ban summary/headnote-only support for core propositions.",
                "scope": "overextension/system.txt",
                "implementation_status": "recommendation_only_do_not_implement",
            }
        )

    mode_d_rejected = rejected_dataframe[rejected_dataframe[MODE_COLUMN] == "D"].copy()
    if not mode_d_rejected.empty:
        recommendations.append(
            {
                "priority": "P2",
                "signal": "mode_D_hierarchy_anchor_gap",
                "evidence_count": int(mode_d_rejected.shape[0]),
                "prompt_recommendation": "Require precedence contracts to include both controlling-authority anchors and non-controlling-status anchors for hierarchy disputes.",
                "scope": "precedence/system.txt",
                "implementation_status": "recommendation_only_do_not_implement",
            }
        )

    if deterministic_counter["expected_citation_outside_packet"] > 0:
        recommendations.append(
            {
                "priority": "P1",
                "signal": "expected_citation_outside_packet",
                "evidence_count": int(deterministic_counter["expected_citation_outside_packet"]),
                "prompt_recommendation": "Reinforce generation prompt self-check to validate every expected citation token exists in the packet before finalizing output.",
                "scope": "generation system prompts (A/C/D)",
                "implementation_status": "recommendation_only_do_not_implement",
            }
        )

    if recommendations:
        return pd.DataFrame(recommendations)

    return pd.DataFrame(
        [
            {
                "priority": "P3",
                "signal": "no_major_prompt_issues_detected",
                "evidence_count": 0,
                "prompt_recommendation": "No high-priority prompt changes detected from this run.",
                "scope": "n/a",
                "implementation_status": "recommendation_only_do_not_implement",
            }
        ]
    )


def write_report(
    output_dir: Path,
    run_timestamp: str,
    packet_ids: list[str],
    validation_dataframe: pd.DataFrame,
    accepted_dataframe: pd.DataFrame,
    rejected_dataframe: pd.DataFrame,
    rejection_by_mode: pd.DataFrame,
    rejection_by_stage: pd.DataFrame,
    risk_counter: Counter[str],
    trace_dataframe: pd.DataFrame,
    recommendations_dataframe: pd.DataFrame,
    max_examples: int,
) -> None:
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

    accepted_reality_dataframe = accepted_dataframe.copy()
    accepted_reality_dataframe["query_length"] = (
        accepted_reality_dataframe["user_query"].astype(str).str.len()
    )
    accepted_reality_dataframe["looks_eval_like"] = accepted_reality_dataframe[
        "user_query"
    ].astype(str).str.contains(
        r"Task:|checklist|closed-world|numbered requirements",
        case=False,
        regex=True,
    )

    query_length_stats = accepted_reality_dataframe["query_length"].describe(
        percentiles=[0.5, 0.9, 0.95]
    )

    flagged_accepted = accepted_reality_dataframe[
        (accepted_reality_dataframe["looks_eval_like"]) 
        | (accepted_reality_dataframe["llm_risk_flag_count"].fillna(0) > 0)
    ].copy()

    trace_status = (
        trace_dataframe.groupby(["stage", "status"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    retry_rows = trace_dataframe[trace_dataframe["attempt"] > 1].copy()

    trace_examples = trace_dataframe[
        [
            "stage",
            "packet",
            "status",
            "attempt",
            "model",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "trace_file",
            "message_prefix",
        ]
    ].copy()

    risk_summary_rows = [
        {"risk_flag": risk_flag, "count": count}
        for risk_flag, count in risk_counter.most_common()
    ]
    risk_summary_dataframe = pd.DataFrame(risk_summary_rows)

    summary_metrics_rows = [
        {"metric": "run_timestamp", "value": run_timestamp},
        {"metric": "packet_count", "value": len(packet_ids)},
        {"metric": "total_items", "value": int(validation_dataframe.shape[0])},
        {"metric": "accepted_items", "value": int(accepted_dataframe.shape[0])},
        {"metric": "rejected_items", "value": int(rejected_dataframe.shape[0])},
        {
            "metric": "accepted_with_llm_risk_flags",
            "value": int(
                (accepted_dataframe["llm_risk_flag_count"].fillna(0) > 0).sum()
            ),
        },
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
    recommendations_dataframe.to_csv(output_dir / "prompt_recommendations.csv", index=False)

    lines: list[str] = []
    lines.append("# Synthetic Generation Audit Report")
    lines.append("")
    lines.append(f"- run_timestamp: `{run_timestamp}`")
    lines.append(f"- packet_ids: `{', '.join(packet_ids)}`")
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

    lines.append("## Rejected Error Modes")
    lines.append(to_markdown_table(rejection_by_mode, max_rows=50))
    lines.append("")
    lines.append("### Rejection Stage")
    lines.append(to_markdown_table(rejection_by_stage, max_rows=50))
    lines.append("")

    lines.append("### Risk Flag Summary (rejected only)")
    lines.append(to_markdown_table(risk_summary_dataframe, max_rows=50))
    lines.append("")

    lines.append("### Rejected Examples")
    rejected_examples = rejected_dataframe[
        [
            "item_id",
            "packet",
            MODE_COLUMN,
            REJECTION_STAGE_COLUMN,
            REJECTION_REASON_COLUMN,
            "user_query",
        ]
    ]
    lines.append(to_markdown_table(rejected_examples, max_rows=max_examples))
    lines.append("")

    lines.append("## Accepted Datapoint Review")
    lines.append(
        f"- accepted items reviewed: `{int(accepted_dataframe.shape[0])}`"
    )
    lines.append(
        f"- accepted items with verifier risk flags: `{int((accepted_dataframe['llm_risk_flag_count'].fillna(0) > 0).sum())}`"
    )
    lines.append(
        f"- accepted items flagged as eval-like by heuristic: `{int(flagged_accepted.shape[0])}`"
    )
    lines.append("")

    lines.append("### Accepted Query Length Stats")
    lines.append(
        to_markdown_table(
            query_length_stats.rename_axis("stat").reset_index(name="value"),
            max_rows=20,
        )
    )
    lines.append("")

    lines.append("### Accepted Query Examples")
    accepted_examples = accepted_dataframe[
        ["item_id", "packet", MODE_COLUMN, "user_query"]
    ]
    lines.append(to_markdown_table(accepted_examples, max_rows=max_examples))
    lines.append("")

    lines.append("## Trace Health")
    lines.append(to_markdown_table(trace_status, max_rows=50))
    lines.append("")
    lines.append(f"- retry trace files (`attempt > 1`): `{int(retry_rows.shape[0])}`")
    lines.append("")

    lines.append("### Trace Examples")
    lines.append(to_markdown_table(trace_examples, max_rows=max_examples))
    lines.append("")

    lines.append("## Prompt Modification Recommendations (Do Not Implement Automatically)")
    lines.append(
        "These are recommendation outputs only. Implementation requires explicit follow-up instruction."
    )
    lines.append("")
    lines.append(to_markdown_table(recommendations_dataframe, max_rows=50))
    lines.append("")

    report_path = output_dir / "analysis_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    rejection_by_mode, rejection_by_stage, risk_counter, deterministic_counter = (
        build_rejection_breakdowns(rejected_dataframe)
    )

    trace_dataframe = collect_trace_health(repo_root, packet_ids, run_timestamp)

    recommendations_dataframe = build_prompt_recommendations(
        rejected_dataframe,
        risk_counter,
        deterministic_counter,
    )

    output_dir = (
        repo_root / "results" / "synthetic_generation_audit" / run_timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    write_report(
        output_dir=output_dir,
        run_timestamp=run_timestamp,
        packet_ids=packet_ids,
        validation_dataframe=validation_dataframe,
        accepted_dataframe=accepted_dataframe,
        rejected_dataframe=rejected_dataframe,
        rejection_by_mode=rejection_by_mode,
        rejection_by_stage=rejection_by_stage,
        risk_counter=risk_counter,
        trace_dataframe=trace_dataframe,
        recommendations_dataframe=recommendations_dataframe,
        max_examples=arguments.max_examples,
    )

    print(f"run_timestamp={run_timestamp}")
    print(f"report={output_dir / 'analysis_summary.md'}")
    print(f"recommendations={output_dir / 'prompt_recommendations.csv'}")


if __name__ == "__main__":
    main()
