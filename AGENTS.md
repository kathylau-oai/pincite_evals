# Pincite Evals — Agent Guidance

This is the repo “house style” for evals and synthetic generation. Follow these rules so runs are **comparable**, **reproducible**, and **easy to debug**.

## Defaults (do these unless you have a strong reason not to)

- **Parallelize safely**: use bounded concurrency (avoid unbounded fan-out).
- **Run apples-to-apples comparisons**: support multiple model configs in one run (model, reasoning effort, temperature, prompt variants) over the exact same inputs.
- **Be reproducible**: write the full configuration used for the run into the run folder (don’t rely on implicit defaults).
- **Use the OpenAI Responses API** for all model calls (do not use Chat Completions).
- **Retry with `tenacity`**: explicit attempt limits, backoff, and a clear set of retriable error types.
- **Progress visibility**: use `tqdm` for long-running drafting/grading jobs.
- **Use DataFrames**: use pandas for load/transform/aggregate; avoid ad-hoc row-by-row bookkeeping.
- **Plotting**: prefer Plotly for slice comparisons and dashboards.

## Secrets

- **Do not ask for an OpenAI API key**. This repo expects credentials via environment variables.

## Outputs (required run artifacts)

Write everything under `results/<experiment>/<run_id>/`.

- **Final artifacts (easy to reload/share)**
  - **Predictions + grades** (CSV)
  - **Aggregated metrics summary** (quality + latency + token usage) with per-model breakdown
  - **Per-slice comparison tables** (per grader and per model)
- **Debug artifacts (make failures diagnosable without reruns)**
  - Raw model responses / traces (when available)
  - Judge / grader intermediate outputs
  - Item-level error annotations (when available)
  - Retry / failure logs + request status metadata

**Spreadsheet friendliness matters**: if an export is intended for Excel, normalize embedded newlines to literal `\n` and quote fields so rows don’t visually “break”.

## Metrics (what to report)

- **Tokens**
  - Token accounting must include **reasoning tokens** when reasoning is enabled.
  - Report distribution stats per model: `avg`, `p50`, `p90`, `p95`, `p99`.
- **Latency**
  - Always report **end-to-end latency** distribution stats per model.
  - **TTFT / inter-token** latency is optional and only applies when a component uses streaming. Do not require streaming just to collect TTFT.

## Analysis quality bar

- **Start grader-first**: compute slice metrics by **grader** first, then compare models.
- **Prefer meaningful slices** when available (citation type, court/jurisdiction, difficulty buckets, domain-specific segments).
- **Make regressions obvious**: clean comparison tables + plots with readable labels and sensible scales.

## Hard contracts (do not drift)

### Citations and packet rendering

- **Only accept dotted packet block IDs**: `DOC###.P###.B##`.
- **Do not accept bracket citation notation** anywhere (items, prompts, normalization, validators, graders).
- **Packet text must have explicit block boundaries** using:
  - `<BLOCK id="DOC###.P###.B##"> ... </BLOCK>`

Enforce these at parse/validation time so “notation-only” differences never become grading disputes.

### Structured outputs and grading contracts

- **Use strict JSON Schema structured outputs** for LLM judges/graders (`text.format` with `strict: true`).
- **Validate required fields at runtime** (at minimum: top-level `passed` plus a non-empty `reason`) before scoring.
- **Do not paper over missing rationale**:
  - Don’t backfill missing trigger notes.
  - Don’t inject fallback caution text.
  - Fail deterministic validation when required fields are missing/empty.

### Grader execution gotchas (easy to regress)

- **Grading from CSV artifacts**: when grading from `predictions.csv` rows, parse `expected_citation_groups_json` / `grading_contract_json` and normalize before building grader context (don’t rely on in-memory dataset fields).
- **Mode A (no required citation groups)**: if `expected_citation_groups` is empty, allow packet-grounded “helpful” citations so expected-citation-presence doesn’t fail by construction.
- **Overextension verdicts**: for categorical labels like `no_overextension` / `overextended`, let the explicit label + `passed` drive the final verdict (don’t override with score-threshold fallbacks).
- **Citation-fidelity short-circuits**: rows like `no_citations_predicted` may not include `judge_result`; validate judge fields only when a judge call actually happened.
- **Reviewer-facing reasons**: when extracting grader reasons, check both top-level `reason` and nested `judge_result.reason` (common nesting pattern).

### Prompt templating

- Use Jinja with `StrictUndefined` (missing placeholders should fail fast).
- Keep instruction-heavy guidance in `system.txt`; keep `user.txt` minimal (task + packet corpus).

## Reliability and throughput

- **Handle local schema/parse failures**: `client.responses.parse(...)` can raise a local `ValidationError` when outputs are truncated/malformed; downgrade to an explicit status instead of crashing a parallel run.
- **Rate limits are normal**: when drafting + grading both use heavy models, reduce `--max-item-workers` and/or `--max-grader-workers` to avoid burst TPM spikes.
- **Avoid late write failures**: resolve output roots to absolute paths and ensure required subdirectories exist immediately before writing artifacts.

## Ergonomics and compatibility (things that keep breaking)

- **Keep ownership consolidated**: runtime code should live under `src/pincite_evals/` (including graders).
- **Eval runner input contract**: prefer a strict `user_query` drafting prompt column; fail fast when missing.
- **Compact exports compatibility** (dashboards/review UIs)
  - Support both `source_user_query` and `user_query` schema variants.
  - Discover graders from any `grader_<name>_<suffix>` columns (`status`, `passed`, `score`, `label`, `reason`, `details_json`).
  - When filtering review tables, select rows by positional `iloc` on the filtered frame to avoid index/label mismatches.

## Synthetic generation (high-signal constraints)

- Validate candidates against the **full packet corpus** (scan all documents for counterevidence).
- Avoid “placeholder” datapoints: retry malformed generation outputs up to the configured attempts; if still malformed, **drop the datapoint** rather than emitting stubs.
- Avoid avoidable incompletes: don’t use hard `max_output_tokens` caps for generation/verifiers; rely on prompt constraints + strict schema parsing.

## Streaming notes (only where applicable)

- Drafting evals typically use `client.responses.create(...)` and do **not** need streaming telemetry (don’t emit TTFT/inter-token columns/artifacts for drafting by default).
- If a component does use streaming, handle terminal `response.incomplete` explicitly (including `content_filter`) and preserve partial output/status for auditability.
