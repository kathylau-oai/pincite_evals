import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import ModeCountConfig, SyntheticGenerationConfig
from .schema import SyntheticItem


CITATION_TOKEN_PATTERN = re.compile(r"^DOC\d{3}\[P\d{3}\.B\d{2}\]$")
MODE_TO_ERROR = {
    "overextension": "C",
    "precedence": "D",
    "fake_citations": "A",
}
ERROR_TO_MODE = {value: key for key, value in MODE_TO_ERROR.items()}
QUALIFIER_TERMS = {
    "may",
    "generally",
    "usually",
    "sometimes",
    "unless",
    "except",
    "however",
    "not",
    "must",
    "only",
    "if",
    "when",
}
PRECEDENCE_TERMS = {
    "overrule",
    "overruled",
    "vacated",
    "vacate",
    "en banc",
    "controlling",
    "binding",
    "persuasive",
    "precedent",
    "circuit",
    "supreme",
}
NEGATION_TERMS = {"not", "however", "unless", "except", "vacated", "overruled"}
STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "where",
    "which",
    "their",
    "there",
    "would",
    "about",
    "were",
    "been",
    "have",
    "shall",
    "could",
    "should",
    "into",
    "your",
    "them",
    "only",
    "also",
    "than",
    "such",
    "each",
    "while",
    "state",
    "court",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']+")


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    target_bank_dir: Path
    raw_candidates_dir: Path
    validation_dir: Path
    selection_dir: Path
    traces_dir: Path


@dataclass(frozen=True)
class PipelineContext:
    config: SyntheticGenerationConfig
    packet_root: Path
    manifest_csv: Path
    blocks_dir: Path
    packet_information_md: Path
    run_paths: RunPaths


@dataclass(frozen=True)
class TargetBankResult:
    target_bank: pd.DataFrame
    sanity_summary: pd.DataFrame


@dataclass(frozen=True)
class GenerationResult:
    candidates_by_mode: dict[str, list[dict[str, Any]]]
    request_metrics: pd.DataFrame


@dataclass(frozen=True)
class ValidationResult:
    accepted_items: list[dict[str, Any]]
    deterministic_pass_items: list[dict[str, Any]]
    deterministic_checks: pd.DataFrame
    llm_reviews: list[dict[str, Any]]
    rejection_log: pd.DataFrame
    request_metrics: pd.DataFrame


@dataclass(frozen=True)
class SelectionResult:
    selected_items: list[dict[str, Any]]
    selection_table: pd.DataFrame
    selection_report_markdown: str


def _compute_distribution_stats(values: pd.Series, prefix: str) -> dict[str, Any]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return {
            f"{prefix}_avg": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
        }

    return {
        f"{prefix}_avg": float(numeric_values.mean()),
        f"{prefix}_p50": float(numeric_values.quantile(0.50)),
        f"{prefix}_p90": float(numeric_values.quantile(0.90)),
        f"{prefix}_p95": float(numeric_values.quantile(0.95)),
        f"{prefix}_p99": float(numeric_values.quantile(0.99)),
    }


def summarize_request_metrics(metrics_dataframe: pd.DataFrame) -> pd.DataFrame:
    if metrics_dataframe.empty:
        return pd.DataFrame([{"stage": "none", "request_count": 0, "completed_count": 0, "items_per_minute": None}])

    summary_rows: list[dict[str, Any]] = []
    for stage_name, stage_rows in metrics_dataframe.groupby("stage"):
        completed_rows = stage_rows[stage_rows["status"] == "completed"]
        total_latency_seconds = pd.to_numeric(completed_rows["latency_seconds"], errors="coerce").dropna().sum()
        completed_count = int(completed_rows.shape[0])
        items_per_minute = None
        if total_latency_seconds > 0 and completed_count > 0:
            items_per_minute = float((completed_count / total_latency_seconds) * 60.0)

        summary_row = {
            "stage": stage_name,
            "request_count": int(stage_rows.shape[0]),
            "completed_count": completed_count,
            "items_per_minute": items_per_minute,
        }
        summary_row.update(_compute_distribution_stats(stage_rows["latency_seconds"], "latency_seconds"))
        summary_row.update(_compute_distribution_stats(stage_rows["input_tokens"], "input_tokens"))
        summary_row.update(_compute_distribution_stats(stage_rows["output_tokens"], "output_tokens"))
        summary_row.update(_compute_distribution_stats(stage_rows["reasoning_tokens"], "reasoning_tokens"))
        summary_row.update(_compute_distribution_stats(stage_rows["total_tokens"], "total_tokens"))
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def _normalize_text(raw_text: str) -> str:
    return " ".join(raw_text.split())


def _extract_terms(text_value: str) -> set[str]:
    return {term.lower() for term in TOKEN_PATTERN.findall(text_value)}


def _extract_keywords(text_value: str) -> set[str]:
    terms = _extract_terms(text_value)
    return {term for term in terms if len(term) >= 6 and term not in STOPWORDS}


def _extract_qualifiers(text_value: str) -> list[str]:
    text_terms = _extract_terms(text_value)
    return [term for term in sorted(QUALIFIER_TERMS) if term in text_terms]


def _create_run_paths(output_root: Path, packet_id: str, run_id: str | None) -> RunPaths:
    resolved_run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / packet_id / resolved_run_id

    target_bank_dir = run_root / "target_bank"
    raw_candidates_dir = run_root / "raw_candidates"
    validation_dir = run_root / "validation"
    selection_dir = run_root / "selection"
    traces_dir = run_root / "traces"

    for folder in [target_bank_dir, raw_candidates_dir, validation_dir, selection_dir, traces_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_root=run_root,
        target_bank_dir=target_bank_dir,
        raw_candidates_dir=raw_candidates_dir,
        validation_dir=validation_dir,
        selection_dir=selection_dir,
        traces_dir=traces_dir,
    )


def _load_packet_manifest(manifest_csv: Path) -> pd.DataFrame:
    return pd.read_csv(manifest_csv)


def _load_packet_blocks(blocks_dir: Path) -> pd.DataFrame:
    block_paths = sorted(blocks_dir.glob("*.blocks.csv"))
    if not block_paths:
        raise ValueError(f"No .blocks.csv files found in {blocks_dir}")
    frames = [pd.read_csv(path) for path in block_paths]
    return pd.concat(frames, ignore_index=True)


def _extract_authority_edges(packet_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    grouped = packet_manifest.groupby("source_filename", dropna=False)
    for source_filename, group in grouped:
        if pd.isna(source_filename):
            continue
        if len(group) <= 1:
            continue
        ordered = group.sort_values("source_order")
        higher_doc_id = str(ordered.iloc[0]["doc_id"])
        for row_index in range(1, len(ordered)):
            rows.append(
                {
                    "lower_doc_id": str(ordered.iloc[row_index]["doc_id"]),
                    "higher_doc_id": higher_doc_id,
                    "relationship": "same_docket_newer_version",
                    "source": "manifest_filename_group",
                }
            )
    return pd.DataFrame(rows)


def _build_counterevidence_lookup(packet_blocks: pd.DataFrame) -> dict[str, list[tuple[str, set[str], str]]]:
    lookup: dict[str, list[tuple[str, set[str], str]]] = {}
    for _, row in packet_blocks.iterrows():
        doc_id = str(row["doc_id"])
        citation_token = str(row["citation_token"])
        block_text = _normalize_text(str(row["text"]))
        block_keywords = _extract_keywords(block_text)
        lookup.setdefault(doc_id, []).append((citation_token, block_keywords, block_text.lower()))
    return lookup


def _find_counterevidence_tokens(row_doc_id: str, row_text: str, lookup: dict[str, list[tuple[str, set[str], str]]]) -> list[str]:
    query_keywords = _extract_keywords(row_text)
    if not query_keywords:
        return []

    matched_tokens: list[str] = []
    for doc_id, blocks in lookup.items():
        if doc_id == row_doc_id:
            continue

        for citation_token, block_keywords, block_text_lower in blocks:
            if len(query_keywords & block_keywords) < 2:
                continue
            if not any(negation_term in block_text_lower for negation_term in NEGATION_TERMS):
                continue

            matched_tokens.append(citation_token)
            if len(matched_tokens) >= 3:
                return matched_tokens

    return matched_tokens


def _build_target_bank_rows(packet_id: str, packet_blocks: pd.DataFrame, edge_doc_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in packet_blocks.iterrows():
        block_text = _normalize_text(str(row["text"]))
        if not block_text:
            continue

        doc_id = str(row["doc_id"])
        citation_token = str(row["citation_token"])
        qualifiers = _extract_qualifiers(block_text)
        precedence_hits = [term for term in PRECEDENCE_TERMS if term in block_text.lower()]

        if qualifiers and len(block_text.split()) >= 12:
            rows.append(
                {
                    "packet_id": packet_id,
                    "mode_name": "overextension",
                    "target_error_mode": "C",
                    "doc_id": doc_id,
                    "citation_token": citation_token,
                    "claim_candidate": block_text,
                    "mode_reason": "qualifier_rich_language",
                    "qualifier_terms": "|".join(qualifiers),
                    "authority_edge_type": "",
                }
            )

        if precedence_hits or doc_id in edge_doc_ids:
            rows.append(
                {
                    "packet_id": packet_id,
                    "mode_name": "precedence",
                    "target_error_mode": "D",
                    "doc_id": doc_id,
                    "citation_token": citation_token,
                    "claim_candidate": block_text,
                    "mode_reason": "precedence_signal" if precedence_hits else "authority_edge_doc",
                    "qualifier_terms": "|".join(sorted(precedence_hits)),
                    "authority_edge_type": "same_docket_newer_version" if doc_id in edge_doc_ids else "",
                }
            )

        if len(block_text.split()) >= 18:
            rows.append(
                {
                    "packet_id": packet_id,
                    "mode_name": "fake_citations",
                    "target_error_mode": "A",
                    "doc_id": doc_id,
                    "citation_token": citation_token,
                    "claim_candidate": block_text,
                    "mode_reason": "substantial_proposition_text",
                    "qualifier_terms": "",
                    "authority_edge_type": "",
                }
            )

    if not rows:
        raise ValueError("No target bank rows were generated.")

    target_bank = pd.DataFrame(rows)
    target_bank = target_bank.drop_duplicates(subset=["mode_name", "citation_token", "mode_reason"]).reset_index(drop=True)

    counterevidence_lookup = _build_counterevidence_lookup(packet_blocks)
    counterevidence_tokens: list[str] = []
    for _, row in target_bank.iterrows():
        tokens = _find_counterevidence_tokens(str(row["doc_id"]), str(row["claim_candidate"]), counterevidence_lookup)
        counterevidence_tokens.append("|".join(tokens))

    target_bank["counterevidence_tokens"] = counterevidence_tokens
    target_bank["counterevidence_count"] = target_bank["counterevidence_tokens"].apply(
        lambda value: 0 if value == "" else len(value.split("|"))
    )
    target_bank["target_id"] = [f"tb_{index:05d}" for index in range(1, len(target_bank) + 1)]

    return target_bank


def _target_bank_sanity(target_bank: pd.DataFrame) -> pd.DataFrame:
    invalid_tokens = target_bank["citation_token"].fillna("").apply(lambda value: CITATION_TOKEN_PATTERN.match(value) is None)
    rows = [
        {"metric": "row_count", "value": int(target_bank.shape[0])},
        {"metric": "column_count", "value": int(target_bank.shape[1])},
        {"metric": "duplicate_rows", "value": int(target_bank.duplicated().sum())},
        {"metric": "missing_claim_candidate", "value": int(target_bank["claim_candidate"].isna().sum())},
        {"metric": "invalid_citation_tokens", "value": int(invalid_tokens.sum())},
    ]
    for mode_name, count in target_bank.groupby("mode_name")["target_id"].count().to_dict().items():
        rows.append({"metric": f"mode_count_{mode_name}", "value": int(count)})
    return pd.DataFrame(rows)


def _build_generation_request(
    model: str,
    reasoning_effort: str,
    temperature: float | None,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort == "none" and temperature is not None:
        request["temperature"] = temperature
    return request


@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
)
def _call_openai_response(client: OpenAI, request: dict[str, Any]) -> Any:
    return client.responses.create(**request)


def _safe_json_object(raw_text: str) -> dict[str, Any]:
    text_value = raw_text.strip()
    parsed = json.loads(text_value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from model output.")
    return parsed


def _normalize_generated_item(
    parsed_candidate: dict[str, Any],
    *,
    mode_name: str,
    citation_token: str,
    packet_id: str,
    as_of_date: str,
) -> dict[str, Any]:
    normalized = dict(parsed_candidate)

    normalized["schema_version"] = "v1"
    normalized["packet_id"] = packet_id
    normalized["as_of_date"] = as_of_date

    raw_mode_value = str(normalized.get("target_error_mode", "")).strip()
    if raw_mode_value in {"A", "C", "D"}:
        normalized["target_error_mode"] = raw_mode_value
    elif raw_mode_value.lower() in MODE_TO_ERROR:
        normalized["target_error_mode"] = MODE_TO_ERROR[raw_mode_value.lower()]
    else:
        normalized["target_error_mode"] = MODE_TO_ERROR[mode_name]

    scenario_facts = normalized.get("scenario_facts")
    if isinstance(scenario_facts, str):
        normalized["scenario_facts"] = [scenario_facts]
    elif not isinstance(scenario_facts, list):
        normalized["scenario_facts"] = ["Assume closed-world packet constraints."]

    grading_contract = normalized.get("grading_contract")
    if not isinstance(grading_contract, dict):
        grading_contract = {}
    normalized["grading_contract"] = grading_contract

    citation_groups = grading_contract.get("expected_citation_groups")
    if isinstance(citation_groups, list):
        cleaned_groups: list[list[str]] = []
        for group in citation_groups:
            if isinstance(group, list):
                cleaned = [str(token).strip() for token in group if str(token).strip()]
                if cleaned:
                    cleaned_groups.append(cleaned)
            elif isinstance(group, str) and group.strip():
                cleaned_groups.append([group.strip()])
        if cleaned_groups:
            grading_contract["expected_citation_groups"] = cleaned_groups
        else:
            grading_contract["expected_citation_groups"] = [[citation_token]]
    else:
        grading_contract["expected_citation_groups"] = [[citation_token]]

    over_note = grading_contract.get("overextension_trigger_note")
    if mode_name == "overextension":
        if not isinstance(over_note, str) or not over_note.strip():
            grading_contract["overextension_trigger_note"] = (
                "Check whether qualified source language is overstated into categorical claims."
            )
    else:
        if "overextension_trigger_note" not in grading_contract:
            grading_contract["overextension_trigger_note"] = None

    precedence_note = grading_contract.get("precedence_trigger_note")
    if mode_name == "precedence":
        if not isinstance(precedence_note, str) or not precedence_note.strip():
            grading_contract["precedence_trigger_note"] = (
                "Check whether controlling authority is used over superseded or non-binding sources."
            )
    else:
        if "precedence_trigger_note" not in grading_contract:
            grading_contract["precedence_trigger_note"] = None

    over_cautions = grading_contract.get("overextension_cautions")
    if not isinstance(over_cautions, list):
        grading_contract["overextension_cautions"] = []

    precedence_cautions = grading_contract.get("precedence_cautions")
    if not isinstance(precedence_cautions, list):
        grading_contract["precedence_cautions"] = []

    if not isinstance(normalized.get("prompt"), str) or not str(normalized.get("prompt", "")).strip():
        normalized["prompt"] = (
            "Draft an internal legal memo section using only packet authorities and cite packet tokens only."
        )

    if not isinstance(normalized.get("item_id"), str):
        normalized["item_id"] = ""
    if not isinstance(normalized.get("query_id"), str):
        normalized["query_id"] = ""

    return normalized


def _mode_system_prompt(mode_name: str) -> str:
    if mode_name == "overextension":
        return (
            "Generate one closed-world legal memo eval item for overextension. "
            "Return JSON only and keep the trap objective."
        )
    if mode_name == "precedence":
        return (
            "Generate one closed-world legal memo eval item for authority hierarchy and precedence. "
            "Return JSON only and keep controlling authority explicit."
        )
    return (
        "Generate one closed-world legal memo eval item for fake citation pressure. "
        "Return JSON only and ensure a compliant answer is possible using packet citations."
    )


def _mode_user_prompt(target_row: pd.Series, packet_id: str, as_of_date: str) -> str:
    return (
        "Return one JSON object with fields: schema_version,item_id,packet_id,target_error_mode,query_id,as_of_date,prompt,scenario_facts,grading_contract. "
        "Use citation tokens in DOC_ID[P###.B##] format. "
        f"packet_id={packet_id}; as_of_date={as_of_date}; mode={target_row['mode_name']}; "
        f"citation_token={target_row['citation_token']}; claim_candidate={str(target_row['claim_candidate'])[:1200]}"
    )


def _default_candidate_from_row(target_row: pd.Series, packet_id: str, as_of_date: str, item_index: int) -> dict[str, Any]:
    mode_name = str(target_row["mode_name"])
    mode_letter = MODE_TO_ERROR[mode_name]
    citation_token = str(target_row["citation_token"])

    over_note = None
    precedence_note = None
    if mode_name == "overextension":
        over_note = "Checks whether qualified source language is overgeneralized into categorical claims."
    if mode_name == "precedence":
        precedence_note = "Checks whether the answer follows controlling authority and rejects superseded authority."

    candidate = {
        "schema_version": "v1",
        "item_id": f"{packet_id}_{mode_letter}_{item_index:02d}",
        "packet_id": packet_id,
        "target_error_mode": mode_letter,
        "query_id": f"q_{mode_name[:3]}_{item_index:04d}",
        "as_of_date": as_of_date,
        "prompt": (
            "Draft an internal legal memo section using only packet authorities. "
            f"Ground analysis in {citation_token} and avoid unsupported categorical statements."
        ),
        "scenario_facts": [
            "Assume a closed-world packet with no external authorities.",
            f"Primary support token: {citation_token}.",
            "Address uncertainty explicitly and avoid fabricated citations.",
        ],
        "grading_contract": {
            "expected_citation_groups": [[citation_token]],
            "overextension_trigger_note": over_note,
            "overextension_cautions": ["Do not reward dropped qualifiers or modal inflation."] if over_note else [],
            "precedence_trigger_note": precedence_note,
            "precedence_cautions": ["Do not treat vacated or persuasive-only authority as controlling."] if precedence_note else [],
        },
    }

    return SyntheticItem.model_validate(candidate).model_dump()


def _generate_one_item(
    *,
    target_row: pd.Series,
    config: SyntheticGenerationConfig,
    item_index: int,
    traces_dir: Path,
    openai_client: OpenAI | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode_name = str(target_row["mode_name"])
    start_time = time.perf_counter()

    if config.dry_run:
        candidate = _default_candidate_from_row(target_row, config.packet_id, config.as_of_date, item_index)
        latency = time.perf_counter() - start_time
        return candidate, {
            "stage": "generation",
            "mode_name": mode_name,
            "target_id": str(target_row["target_id"]),
            "status": "completed",
            "latency_seconds": latency,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    if openai_client is None:
        raise ValueError("OpenAI client is required when dry_run is false.")

    request = _build_generation_request(
        model=config.generation_model,
        reasoning_effort=config.generation_reasoning_effort,
        temperature=config.generation_temperature,
        system_prompt=_mode_system_prompt(mode_name),
        user_prompt=_mode_user_prompt(target_row, packet_id=config.packet_id, as_of_date=config.as_of_date),
        max_output_tokens=1800,
    )

    response = _call_openai_response(openai_client, request)
    latency = time.perf_counter() - start_time

    if response.status == "incomplete" and response.incomplete_details and response.incomplete_details.reason == "max_output_tokens":
        if not response.output_text:
            validated_candidate = _default_candidate_from_row(target_row, config.packet_id, config.as_of_date, item_index)
            usage = response.usage
            return validated_candidate, {
                "stage": "generation",
                "mode_name": mode_name,
                "target_id": str(target_row["target_id"]),
                "status": "incomplete_fallback_default_candidate",
                "latency_seconds": latency,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "reasoning_tokens": usage.output_tokens_details.reasoning_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            }

    try:
        parsed_candidate = _safe_json_object(response.output_text)
        normalized_candidate = _normalize_generated_item(
            parsed_candidate,
            mode_name=mode_name,
            citation_token=str(target_row["citation_token"]),
            packet_id=config.packet_id,
            as_of_date=config.as_of_date,
        )
        validated_candidate = SyntheticItem.model_validate(normalized_candidate).model_dump()
        generation_status = response.status
    except (json.JSONDecodeError, ValueError):
        # Fallback keeps the run moving when model JSON is malformed.
        validated_candidate = _default_candidate_from_row(target_row, config.packet_id, config.as_of_date, item_index)
        generation_status = "fallback_default_candidate"
    mode_letter = MODE_TO_ERROR[mode_name]
    validated_candidate["item_id"] = f"{config.packet_id}_{mode_letter}_{item_index:02d}"
    validated_candidate["query_id"] = f"q_{mode_name[:3]}_{item_index:04d}"

    trace_path = traces_dir / "raw_generation_responses" / f"{mode_name}_{target_row['target_id']}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    usage = response.usage
    return validated_candidate, {
        "stage": "generation",
        "mode_name": mode_name,
        "target_id": str(target_row["target_id"]),
        "status": generation_status,
        "latency_seconds": latency,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }


def _deterministic_validation(
    item_payload: dict[str, Any],
    citation_universe: set[str],
    counterevidence_map: dict[str, list[str]],
    edge_doc_ids: set[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    item = SyntheticItem.model_validate(item_payload)

    expected_citations = [
        citation_token
        for citation_group in item.grading_contract.expected_citation_groups
        for citation_token in citation_group
    ]

    for citation_token in expected_citations:
        if citation_token not in citation_universe:
            reasons.append("expected_citation_outside_packet")

        if counterevidence_map.get(citation_token):
            reasons.append("holding_vs_dicta_mismatch")

        citation_doc_id = citation_token.split("[")[0]
        if item.target_error_mode == "D" and citation_doc_id in edge_doc_ids and counterevidence_map.get(citation_token):
            reasons.append("same_name_case_misattribution")

    if item.target_error_mode == "C" and not item.grading_contract.overextension_trigger_note:
        reasons.append("missing_overextension_trigger_note")

    if item.target_error_mode == "D" and not item.grading_contract.precedence_trigger_note:
        reasons.append("missing_precedence_trigger_note")

    deduped_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for reason in reasons:
        if reason in seen_reasons:
            continue
        deduped_reasons.append(reason)
        seen_reasons.add(reason)

    return len(deduped_reasons) == 0, deduped_reasons, {
        "expected_citation_count": len(expected_citations),
        "counterevidence_matches": sum(1 for token in expected_citations if counterevidence_map.get(token)),
    }


def _build_validation_request(config: SyntheticGenerationConfig, item_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.validation_model,
        "instructions": (
            "Evaluate this synthetic legal eval item for objective consensus grading quality. "
            "Assume citation-token references point to real packet blocks and do not fail only because raw packet text is not embedded in the item JSON. "
            "Focus on ambiguity, internal consistency, and whether grading guidance is specific enough for reliable pass/fail judgment. "
            "Return JSON only with keys verdict, reason, risk_flags, suggested_fix. verdict must be pass or fail."
        ),
        "input": [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(item_payload, ensure_ascii=True)}]}],
        "reasoning": {"effort": config.validation_reasoning_effort},
        "max_output_tokens": 600,
    }


def _validate_one_item(
    *,
    config: SyntheticGenerationConfig,
    item_payload: dict[str, Any],
    traces_dir: Path,
    openai_client: OpenAI | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_time = time.perf_counter()

    if config.dry_run:
        return {
            "item_id": item_payload["item_id"],
            "verdict": "pass",
            "reason": "Dry-run verifier pass.",
            "risk_flags": [],
            "suggested_fix": "",
            "full_response": {"dry_run": True},
        }, {
            "stage": "validation",
            "item_id": item_payload["item_id"],
            "status": "completed",
            "latency_seconds": time.perf_counter() - start_time,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "total_tokens": None,
        }

    if openai_client is None:
        raise ValueError("OpenAI client is required when dry_run is false.")

    request = _build_validation_request(config, item_payload)
    response = _call_openai_response(openai_client, request)
    latency = time.perf_counter() - start_time

    try:
        parsed = _safe_json_object(response.output_text)
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "fail"}:
            verdict = "fail"
            parsed = {
                "reason": "Verifier returned invalid verdict format.",
                "risk_flags": ["invalid_verifier_output"],
                "suggested_fix": "Return JSON with verdict=pass|fail.",
            }
    except (json.JSONDecodeError, ValueError):
        verdict = "fail"
        parsed = {
            "reason": "Verifier response was not valid JSON.",
            "risk_flags": ["invalid_verifier_output"],
            "suggested_fix": "Ensure verifier returns strict JSON object.",
        }

    review = {
        "item_id": item_payload["item_id"],
        "verdict": verdict,
        "reason": str(parsed.get("reason", "")).strip(),
        "risk_flags": parsed.get("risk_flags", []),
        "suggested_fix": str(parsed.get("suggested_fix", "")).strip(),
        "full_response": response.model_dump(),
    }

    trace_path = traces_dir / "raw_validation_responses" / f"{item_payload['item_id']}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    usage = response.usage
    metrics = {
        "stage": "validation",
        "item_id": item_payload["item_id"],
        "status": response.status,
        "latency_seconds": latency,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }
    return review, metrics


def _quality_score(item_payload: dict[str, Any]) -> float:
    prompt_score = min(len(item_payload["prompt"].split()), 200) / 200.0
    fact_score = min(len(item_payload.get("scenario_facts", [])), 5) / 5.0
    citation_group_score = min(len(item_payload["grading_contract"]["expected_citation_groups"]), 3) / 3.0
    return prompt_score + fact_score + citation_group_score


def _trap_signature(prompt_text: str) -> str:
    words = [word.strip(".,;:()[]{}").lower() for word in prompt_text.split()]
    return " ".join(words[:14])


def _primary_doc_id(item_payload: dict[str, Any]) -> str:
    first_citation = item_payload["grading_contract"]["expected_citation_groups"][0][0]
    return first_citation.split("[")[0]


def _select_mode_items(mode_rows: pd.DataFrame, keep_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_signatures: set[str] = set()
    used_docs: set[str] = set()

    for _, row in mode_rows.sort_values("quality_score", ascending=False).iterrows():
        signature = str(row["trap_signature"])
        doc_id = str(row["primary_doc_id"])

        if signature in used_signatures:
            continue
        if doc_id in used_docs and len(used_docs) < keep_count:
            continue

        selected.append(row["payload"])
        used_signatures.add(signature)
        used_docs.add(doc_id)
        if len(selected) == keep_count:
            break

    if len(selected) < keep_count:
        for _, row in mode_rows.sort_values("quality_score", ascending=False).iterrows():
            if row["payload"] in selected:
                continue
            selected.append(row["payload"])
            if len(selected) == keep_count:
                break

    return selected


class SyntheticGenerationPipeline:
    def __init__(self, config: SyntheticGenerationConfig):
        self.config = config

    def bootstrap(self, run_id: str | None = None) -> PipelineContext:
        packet_root = self.config.packet_root / self.config.packet_id
        manifest_csv = packet_root / "packet_manifest.csv"
        blocks_dir = packet_root / "blocks"
        packet_information_md = packet_root / "packet_information.md"

        if not packet_root.exists():
            raise ValueError(f"Packet folder not found: {packet_root}")
        if not manifest_csv.exists():
            raise ValueError(f"Packet manifest not found: {manifest_csv}")

        run_paths = _create_run_paths(self.config.output_root, self.config.packet_id, run_id)
        (run_paths.run_root / "config_snapshot.md").write_text(
            "\n".join(
                [
                    f"packet_id: {self.config.packet_id}",
                    f"generation_model: {self.config.generation_model}",
                    f"generation_reasoning_effort: {self.config.generation_reasoning_effort}",
                    f"validation_model: {self.config.validation_model}",
                    f"validation_reasoning_effort: {self.config.validation_reasoning_effort}",
                    f"dry_run: {self.config.dry_run}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        return PipelineContext(
            config=self.config,
            packet_root=packet_root,
            manifest_csv=manifest_csv,
            blocks_dir=blocks_dir,
            packet_information_md=packet_information_md,
            run_paths=run_paths,
        )

    def run_target_bank(self, context: PipelineContext) -> TargetBankResult:
        manifest = _load_packet_manifest(context.manifest_csv)
        blocks = _load_packet_blocks(context.blocks_dir)
        authority_edges = _extract_authority_edges(manifest)
        edge_doc_ids = set(authority_edges["lower_doc_id"].astype(str).tolist()) | set(
            authority_edges["higher_doc_id"].astype(str).tolist()
        )

        target_bank = _build_target_bank_rows(context.config.packet_id, blocks, edge_doc_ids)
        sanity = _target_bank_sanity(target_bank)

        target_bank.to_csv(context.run_paths.target_bank_dir / "target_bank.csv", index=False)
        sanity.to_csv(context.run_paths.target_bank_dir / "sanity_checks.csv", index=False)
        authority_edges.to_csv(context.run_paths.target_bank_dir / "authority_edges.csv", index=False)

        # Guardrail: ensure each required mode exists before generation.
        counts = target_bank.groupby("mode_name")["target_id"].count().to_dict()
        for required_mode in ["overextension", "precedence", "fake_citations"]:
            if counts.get(required_mode, 0) == 0:
                raise ValueError(f"Target bank has no rows for mode {required_mode}")

        return TargetBankResult(target_bank=target_bank, sanity_summary=sanity)

    def run_generation(
        self,
        *,
        context: PipelineContext,
        target_bank: pd.DataFrame,
        openai_client: OpenAI | None = None,
    ) -> GenerationResult:
        per_mode_counts = {
            "overextension": context.config.generate_count.overextension,
            "precedence": context.config.generate_count.precedence,
            "fake_citations": context.config.generate_count.fake_citations,
        }

        candidates_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in per_mode_counts}
        metrics_rows: list[dict[str, Any]] = []

        def generate_mode(mode_name: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
            mode_targets = (
                target_bank[target_bank["mode_name"] == mode_name]
                .sort_values(["counterevidence_count", "target_id"], ascending=[True, True])
                .head(per_mode_counts[mode_name])
            )
            if mode_targets.empty:
                raise ValueError(f"No targets available for mode {mode_name}")

            mode_candidates: list[dict[str, Any]] = []
            mode_metrics: list[dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=min(context.config.parallelism.generation_workers, len(mode_targets))) as item_pool:
                futures = {}
                for item_index, (_, row) in enumerate(mode_targets.iterrows(), start=1):
                    futures[
                        item_pool.submit(
                            _generate_one_item,
                            target_row=row,
                            config=context.config,
                            item_index=item_index,
                            traces_dir=context.run_paths.traces_dir,
                            openai_client=openai_client,
                        )
                    ] = str(row["target_id"])

                for future in as_completed(futures):
                    target_id = futures[future]
                    try:
                        candidate, metric = future.result()
                    except OpenAIError as error:
                        mode_metrics.append(
                            {
                                "stage": "generation",
                                "mode_name": mode_name,
                                "target_id": target_id,
                                "status": "error",
                                "latency_seconds": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "reasoning_tokens": None,
                                "total_tokens": None,
                                "error_type": type(error).__name__,
                            }
                        )
                        continue

                    mode_candidates.append(candidate)
                    mode_metrics.append(metric)

            return mode_name, mode_candidates, mode_metrics

        with ThreadPoolExecutor(max_workers=min(context.config.parallelism.mode_workers, len(per_mode_counts))) as mode_pool:
            mode_futures = [mode_pool.submit(generate_mode, mode_name) for mode_name in per_mode_counts]
            for mode_future in as_completed(mode_futures):
                mode_name, mode_candidates, mode_metrics = mode_future.result()
                candidates_by_mode[mode_name] = mode_candidates
                metrics_rows.extend(mode_metrics)

        metrics_df = pd.DataFrame(metrics_rows)
        for mode_name, candidates in candidates_by_mode.items():
            with (context.run_paths.raw_candidates_dir / f"{mode_name}_candidates.jsonl").open("w", encoding="utf-8") as file_handle:
                for candidate in candidates:
                    file_handle.write(json.dumps(candidate, ensure_ascii=True) + "\n")

        metrics_df.to_csv(context.run_paths.traces_dir / "generation_metrics.csv", index=False)
        return GenerationResult(candidates_by_mode=candidates_by_mode, request_metrics=metrics_df)

    def run_validation(
        self,
        *,
        context: PipelineContext,
        target_bank: pd.DataFrame,
        candidates: list[dict[str, Any]],
        openai_client: OpenAI | None = None,
    ) -> ValidationResult:
        citation_universe = set(target_bank["citation_token"].dropna().astype(str).tolist())
        counterevidence_map = {
            str(row["citation_token"]): (
                str(row["counterevidence_tokens"]).split("|") if str(row["counterevidence_tokens"]) else []
            )
            for _, row in target_bank.iterrows()
        }
        edge_doc_ids = set(target_bank[target_bank["authority_edge_type"] != ""]["doc_id"].astype(str).tolist())

        deterministic_rows: list[dict[str, Any]] = []
        rejection_rows: list[dict[str, Any]] = []
        items_for_llm: list[dict[str, Any]] = []

        for item in candidates:
            deterministic_pass, reasons, details = _deterministic_validation(
                item,
                citation_universe,
                counterevidence_map,
                edge_doc_ids,
            )
            deterministic_rows.append(
                {
                    "item_id": item["item_id"],
                    "deterministic_pass": deterministic_pass,
                    "reason_codes": "|".join(reasons),
                    "expected_citation_count": details["expected_citation_count"],
                    "counterevidence_matches": details["counterevidence_matches"],
                }
            )
            if deterministic_pass:
                items_for_llm.append(item)
            else:
                rejection_rows.append(
                    {
                        "item_id": item["item_id"],
                        "rejection_stage": "deterministic",
                        "rejection_reason": "|".join(reasons),
                    }
                )

        llm_reviews: list[dict[str, Any]] = []
        accepted_items: list[dict[str, Any]] = []
        metrics_rows: list[dict[str, Any]] = []

        if items_for_llm:
            with ThreadPoolExecutor(max_workers=min(context.config.parallelism.validation_workers, len(items_for_llm))) as pool:
                futures = {
                    pool.submit(
                        _validate_one_item,
                        config=context.config,
                        item_payload=item,
                        traces_dir=context.run_paths.traces_dir,
                        openai_client=openai_client,
                    ): item
                    for item in items_for_llm
                }

                for future in as_completed(futures):
                    item = futures[future]
                    item_id = item["item_id"]
                    try:
                        review, metrics = future.result()
                    except OpenAIError as error:
                        rejection_rows.append(
                            {
                                "item_id": item_id,
                                "rejection_stage": "llm_verifier",
                                "rejection_reason": type(error).__name__,
                            }
                        )
                        metrics_rows.append(
                            {
                                "stage": "validation",
                                "item_id": item_id,
                                "status": "error",
                                "latency_seconds": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "reasoning_tokens": None,
                                "total_tokens": None,
                            }
                        )
                        continue

                    llm_reviews.append(review)
                    metrics_rows.append(metrics)
                    if review["verdict"] == "pass":
                        accepted_items.append(item)
                    else:
                        rejection_rows.append(
                            {
                                "item_id": item_id,
                                "rejection_stage": "llm_verifier",
                                "rejection_reason": review["reason"],
                            }
                        )

        deterministic_df = pd.DataFrame(deterministic_rows)
        rejection_df = pd.DataFrame(rejection_rows)
        metrics_df = pd.DataFrame(metrics_rows)

        deterministic_df.to_csv(context.run_paths.validation_dir / "deterministic_checks.csv", index=False)
        with (context.run_paths.validation_dir / "llm_consensus_reviews.jsonl").open("w", encoding="utf-8") as file_handle:
            for review in llm_reviews:
                file_handle.write(json.dumps(review, ensure_ascii=True) + "\n")
        rejection_df.to_csv(context.run_paths.validation_dir / "rejection_log.csv", index=False)
        metrics_df.to_csv(context.run_paths.traces_dir / "validation_metrics.csv", index=False)

        return ValidationResult(
            accepted_items=accepted_items,
            deterministic_pass_items=items_for_llm,
            deterministic_checks=deterministic_df,
            llm_reviews=llm_reviews,
            rejection_log=rejection_df,
            request_metrics=metrics_df,
        )

    def run_selection(self, *, context: PipelineContext, accepted_items: list[dict[str, Any]]) -> SelectionResult:
        if not accepted_items:
            raise ValueError("No accepted items available for selection.")

        table_rows: list[dict[str, Any]] = []
        for item in accepted_items:
            table_rows.append(
                {
                    "item_id": item["item_id"],
                    "mode_name": ERROR_TO_MODE[item["target_error_mode"]],
                    "target_error_mode": item["target_error_mode"],
                    "primary_doc_id": _primary_doc_id(item),
                    "trap_signature": _trap_signature(item["prompt"]),
                    "quality_score": _quality_score(item),
                    "payload": item,
                }
            )

        item_table = pd.DataFrame(table_rows)

        keep_counts = {
            "overextension": context.config.final_keep_count.overextension,
            "precedence": context.config.final_keep_count.precedence,
            "fake_citations": context.config.final_keep_count.fake_citations,
        }

        selected_items: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []

        for mode_name, keep_count in keep_counts.items():
            mode_rows = item_table[item_table["mode_name"] == mode_name]
            if mode_rows.empty:
                raise ValueError(f"No accepted items for mode {mode_name}")

            mode_selected = _select_mode_items(mode_rows, keep_count)
            if len(mode_selected) < keep_count:
                raise ValueError(f"Insufficient items for mode {mode_name}: {len(mode_selected)} < {keep_count}")

            selected_items.extend(mode_selected)
            for item in mode_selected:
                selection_rows.append(
                    {
                        "item_id": item["item_id"],
                        "mode_name": mode_name,
                        "target_error_mode": item["target_error_mode"],
                        "primary_doc_id": _primary_doc_id(item),
                    }
                )

        selection_table = pd.DataFrame(selection_rows).sort_values(["mode_name", "item_id"]).reset_index(drop=True)

        report_lines = [
            "# Selection Report",
            "",
            f"Total selected items: {len(selected_items)}",
            "",
            "## Per-mode counts",
        ]
        for mode_name, count in selection_table.groupby("mode_name")["item_id"].count().to_dict().items():
            report_lines.append(f"- {mode_name}: {count}")
        report_lines.append("")
        report_lines.append("## Selected items")
        for _, row in selection_table.iterrows():
            report_lines.append(f"- {row['item_id']} ({row['mode_name']}, {row['primary_doc_id']})")
        report_text = "\n".join(report_lines) + "\n"

        with (context.run_paths.selection_dir / "selected_items.jsonl").open("w", encoding="utf-8") as file_handle:
            for item in selected_items:
                file_handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        selection_table.to_csv(context.run_paths.selection_dir / "selected_items.csv", index=False)
        (context.run_paths.selection_dir / "selection_report.md").write_text(report_text, encoding="utf-8")

        return SelectionResult(
            selected_items=selected_items,
            selection_table=selection_table,
            selection_report_markdown=report_text,
        )

    def export_canonical_dataset(self, selected_items: list[dict[str, Any]]) -> Path:
        dataset_dir = self.config.dataset_root / self.config.packet_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        with (dataset_dir / "synthetic_items.jsonl").open("w", encoding="utf-8") as file_handle:
            for item in selected_items:
                file_handle.write(json.dumps(item, ensure_ascii=True) + "\n")

        pd.DataFrame(selected_items).to_csv(dataset_dir / "synthetic_items.csv", index=False)
        return dataset_dir

    def run_all(self, context: PipelineContext, openai_client: OpenAI | None = None) -> dict[str, Any]:
        target_bank_result = self.run_target_bank(context)
        generation_result = self.run_generation(
            context=context,
            target_bank=target_bank_result.target_bank,
            openai_client=openai_client,
        )

        all_candidates: list[dict[str, Any]] = []
        for mode_items in generation_result.candidates_by_mode.values():
            all_candidates.extend(mode_items)

        validation_result = self.run_validation(
            context=context,
            target_bank=target_bank_result.target_bank,
            candidates=all_candidates,
            openai_client=openai_client,
        )
        items_for_selection = (
            validation_result.accepted_items
            if validation_result.accepted_items
            else validation_result.deterministic_pass_items
        )
        selection_result = self.run_selection(context=context, accepted_items=items_for_selection)
        dataset_dir = self.export_canonical_dataset(selection_result.selected_items)

        metrics_frames = [frame for frame in [generation_result.request_metrics, validation_result.request_metrics] if not frame.empty]
        combined_metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
        metrics_summary = summarize_request_metrics(combined_metrics)
        metrics_summary.to_csv(context.run_paths.traces_dir / "metrics.csv", index=False)

        return {
            "target_bank_rows": int(target_bank_result.target_bank.shape[0]),
            "generated_counts": {mode: len(items) for mode, items in generation_result.candidates_by_mode.items()},
            "accepted_items": len(validation_result.accepted_items),
            "selection_fallback_used": len(validation_result.accepted_items) == 0,
            "selected_items": len(selection_result.selected_items),
            "run_root": str(context.run_paths.run_root),
            "dataset_dir": str(dataset_dir),
        }


def load_candidates_from_run(raw_candidates_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for mode_name in ["overextension", "precedence", "fake_citations"]:
        mode_file = raw_candidates_dir / f"{mode_name}_candidates.jsonl"
        if not mode_file.exists():
            continue
        with mode_file.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                candidates.append(json.loads(line))
    return candidates


def build_openai_client(config: SyntheticGenerationConfig) -> OpenAI:
    return OpenAI(timeout=config.request_timeout_seconds)
