# Pincite Evals — Agent Guidance

This file is the “house style” for running evals in this repo. When adding or modifying evaluation code, follow these constraints so runs are comparable, debuggable, and fast.

## Non-negotiables (do these by default)

- **Parallelize work** whenever possible to reduce end-to-end runtime.
- **Support multiple model configurations in one run** (e.g., model, reasoning effort, temperature, prompt variants) so comparisons are apples-to-apples on identical inputs.
- **Make runs reproducible**: keep configuration explicit and fully recorded in outputs.
- **Use the OpenAI Responses API** for evaluation calls (do not use Chat Completions).
- **Use `tenacity` for retries** with explicit attempt limits, backoff, and a clear definition of retriable error types.
- **Use `tqdm`** progress bars for long-running evaluation and grading jobs.
- **Use pandas DataFrames** for loading, transforming, and aggregating results.
- **Use Plotly** for analysis and plotting outputs, especially for slice comparisons.

## API keys / secrets

- **Do not ask for an OpenAI API key**. This repo already has the key set in environment variables; just run commands that rely on it.

## Results and experiment structure

- **Write all outputs under `results/`**, grouped under an experiment-specific subfolder (one folder per run).
- **Always save “final” artifacts** that are easy to reload/share:
  - Final predictions + grades CSV
  - Aggregated metrics summary (quality + latency + token stats)
  - Per-model and per-slice comparison tables
- **Also save “debuggable” intermediate artifacts** so failures can be diagnosed without re-running:
  - Raw model responses / traces
  - Judge / grader intermediate outputs
  - Item-level error annotations (when available)
  - Retry / failure logs and status metadata

## Analysis quality bar

- **Start grader-first**: compute slice-level metrics broken out by grader and compare grader behavior directly.
- **Report meaningful slices** (when applicable), for example:
  - Citation type
  - Court / jurisdiction
  - Difficulty buckets
  - Other domain-relevant segments
- **Make regressions obvious** with clear visuals and tables.
  - Include bar plots for:
    - Per grader
    - Per model run
    - Per grader × per model run comparisons
  - Ensure plots are polished: clear labels, no truncation/overlap, legible legends, sensible scales.

## Metrics requirements

- **Token accounting must include reasoning tokens** whenever reasoning is enabled. Do not report token usage without them.
- **Token metrics**: report distribution stats (`avg`, `p50`, `p90`, `p95`, `p99`) with a clear per-model breakdown.
- **Latency metrics must include**:
  - End-to-end latency
  - Time-to-first-token (TTFT)
  - Inter-token latency (token-by-token) with distribution stats

## Recent learnings (keep these invariants in mind)

Grouped by theme to make constraints easier to find. Each item captures a concrete problem we hit, the fix, and why it matters.

### Citations, normalization, and packet rendering

- **Excerpt citation parsing accepted overly broad IDs and legacy styles**
  - **Fix**: Enforce only dotted packet block IDs (`DOC###.P###.B##`) in parser/tests and reject legacy paragraph citations.
  - **Why**: Keeps citations block-level, deterministic, and consistent with the current grading format.

- **Packet text needed unambiguous, low-noise block boundaries**
  - **Fix**: Render annotated packet text with XML wrappers: `<BLOCK id="DOC###.P###.B##"> ... </BLOCK>` per block.
  - **Why**: Makes block anchors explicit with lower token noise and simpler parsing.

- **Packet annotations exposed XML block IDs while some components expected bracket citations**
  - **Fix**: Standardize on dotted packet block IDs (`DOC###.P###.B##`) everywhere and reject bracket citation notation.
  - **Why**: Eliminates notation drift and keeps parsing/validation deterministic.

- **Verifier false-rejected due to mixed citation notation between item JSON and packet corpus**
  - **Fix**: Require dotted packet block IDs (`DOC###.P###.B##`) in item payloads and verifier prompts; fail bracket-style citations at validation time.
  - **Why**: Prevents notation-only disputes and keeps validation focused on substantive grading quality.

### Synthetic generation: scope, quality, and throughput

- **Synthetic traps looked valid when only local excerpts were checked**
  - **Fix**: Require generation and validation to scan all packet documents for counterevidence before accepting a candidate.
  - **Why**: Prevents ambiguous items and improves lawyer-consensus reliability.

- **Synthetic pipeline scope drifted toward all error modes**
  - **Fix**: Explicitly scope adversarial generation to `A/C/D` and treat `B` as grader-only measurement for this phase.
  - **Why**: Preserves focus on high-signal trap types while still tracking span errors in downstream grading.

- **Generation/verification was slow and config drifted**
  - **Fix**: Default to `gpt-5.2` with high reasoning for generation + verifier and run mode workers in parallel with bounded concurrency.
  - **Why**: Improves quality consistency and end-to-end throughput.

- **Deterministic conflict checks could over-reject and starve quotas**
  - **Fix**: Prioritize low-counterevidence targets during generation (`counterevidence_count` ascending) before LLM calls.
  - **Why**: Maintains strict validation while keeping enough pass candidates for stable 3/3/3 selection.

- **Deterministic target-bank seeding constrained discovery**
  - **Fix**: Build a full packet corpus (8 docs) and let the model discover trap opportunities directly per mode.
  - **Why**: Aligns generation with prompt-driven adversarial discovery and reduces hand-crafted target bias.

- **Hard `max_output_tokens` caps caused avoidable incompletes**
  - **Fix**: Remove explicit `max_output_tokens` for synthetic generation/verifier calls; rely on prompt constraints + schema parsing.
  - **Why**: Reduces truncation failures while preserving structured-output validation.

- **Mode A hallucination traps were too answerable and over-constrained**
  - **Fix**: Redefine Mode A prompts to request absent authority, allow empty/minimal `expected_citation_groups`, and treat compliant “insufficient packet support” responses as the pass path.
  - **Why**: Better isolates fabricated-citation behavior instead of penalizing valid packet-grounded answers.

- **Failed generation requests emitted placeholder candidates**
  - **Fix**: Retry malformed generation outputs up to configured attempts; then drop the datapoint from candidates instead of emitting fallback items.
  - **Why**: Keeps datasets free of stub noise and preserves quality signal from real model outputs only.

### Pipeline ergonomics and auditability

- **Synthetic pipeline was hard to follow across many files**
  - **Fix**: Consolidate into `config.py`, `schema.py`, `pipeline.py`, and a single `cli.py` with subcommands.
  - **Why**: Keeps ownership simple and reduces navigation overhead during fast iteration.

- **Validation outputs were split and hard to audit end-to-end**
  - **Fix**: Export `validation/llm_consensus_reviews.csv` plus `validation/validation_datapoints.csv` merging candidate payload fields, deterministic checks, LLM verdicts, rejection reasons, and request metrics.
  - **Why**: Enables fast audit of accepted vs rejected datapoints without manual joins.

- **Eval runner input contract should stay strict on `user_query` once datasets are regenerated**
  - **Fix**: Enforce `user_query` as the only drafting prompt column in the eval runner and fail fast when missing.
  - **Why**: Avoids silent schema drift and keeps packet-generation and eval-runner contracts aligned.

- **Runner naming drift (`template_eval_runner` vs `eval_runner`) made discoverability worse**
  - **Fix**: Standardize module/CLI naming on `eval_runner` and `pincite-eval`.
  - **Why**: Keeps entrypoints obvious and reduces maintenance overhead.

- **Root-level `graders/` package created split ownership and import ambiguity**
  - **Fix**: Move graders into `src/pincite_evals/graders` and import via `pincite_evals.graders`.
  - **Why**: Keeps all runtime code under one package tree and avoids path-dependent behavior.

- **Run artifacts failed to write when `generation/candidates` was missing mid-run**
  - **Fix**: Resolve packet/output/dataset roots to absolute paths during config load and re-`mkdir` generation output subdirectories immediately before candidate/metrics writes.
  - **Why**: Prevents late-stage `FileNotFoundError` from cwd drift or partial-run directory cleanup.

- **Synthetic generation + quality audit needed a repeatable one-command path**
  - **Fix**: Use `skills/synthetic-generation-audit/scripts/run_and_analyze.sh` to run (or reuse) all-packet generation and emit a standardized audit report under `results/synthetic_generation_audit/<run_timestamp>/`.
  - **Why**: Keeps accepted/rejected analysis, trace health checks, and prompt-only recommendations consistent across runs.

- **Heuristic rejection labels were easy to over-trust during audit**
  - **Fix**: Treat scripted rejection fields as hints only and require evidence-first review using `rejected_reasoning_evidence.csv`, `accepted_reasoning_evidence.csv`, and `trace_reasoning_evidence.csv`.
  - **Why**: Preserves model-led root-cause reasoning and avoids circular conclusions from pre-labeled metadata.

- **Verifier retry failures were undercounted when relying on trace exports alone**
  - **Fix**: Treat `validation_request_status` in `validation_datapoints.csv` as the source of truth for verifier retry failures (for example `verifier_request_failed_after_4_attempts:RateLimitError`) and use `trace_health.csv` as a completed-trace/latency view.
  - **Why**: Failed verifier calls may not emit trace JSON artifacts, so trace-only audits can hide operational failure rates.

### Reliability and correctness in structured outputs / grading contracts

- **`client.responses.parse(...)` can raise local `ValidationError` on truncated/malformed structured output**
  - **Fix**: Catch parse `ValidationError` in generation/validation workers and downgrade to explicit fallback statuses instead of crashing the run.
  - **Why**: Preserves long parallel jobs and keeps failure modes auditable in metrics.

- **Backfilled trigger-note defaults hid missing model rationale**
  - **Fix**: Do not backfill missing trigger notes in normalization; fail deterministic validation when the expected error-mode trigger note is missing.
  - **Why**: Preserves signal about model completeness and prevents silently passing under-specified grading contracts.

- **Fallback caution injection hid weak model-produced grading contracts**
  - **Fix**: Never inject fallback caution text; keep only model-produced notes/cautions and fail deterministic validation when required note/caution fields are empty.
  - **Why**: Ensures grader rationale quality is attributable to model output and failures remain transparent.

### Latency observability

- **Needed TTFT and inter-token latency from Responses API calls**
  - **Fix**: Use `client.responses.stream(...)`, timestamp `response.output_text.delta` events, and finalize with `stream.get_final_response()` for status/usage.
  - **Why**: Captures latency distributions and token usage (including reasoning tokens) from one request path.

### Prompt templating

- **String `.replace(...)` prompt injection drifted as templates grew**
  - **Fix**: Centralize prompt rendering with Jinja (`Environment(undefined=StrictUndefined)`) and migrate prompt files to explicit Jinja variables.
  - **Why**: Missing placeholders now fail fast, prompt loading is consistent across pipelines, and template changes are safer to refactor.

- **Generation prompts mixed critical instructions into `user` messages**
  - **Fix**: Keep instruction-heavy guidance in mode `system.txt` templates (including `{{ lawyer_query_style_guide }}`) and keep mode `user.txt` prompts minimal (`generate` + packet corpus only).
  - **Why**: Preserves instruction priority and reduces drift where high-priority constraints are buried in lower-priority prompt roles.

### Eval throughput and rate limits

- **Full eval runs hit TPM limits when drafting and graders both used `gpt-5.2` with high concurrency**
  - **Fix**: For all-`gpt-5.2` runs, reduce `--max-item-workers` and/or `--max-grader-workers` to avoid burst token spikes, especially with large packet prompts.
  - **Why**: Prevents large `response_status=error` / `skipped_model_error` cohorts that invalidate quality comparisons.

### Grader context and contracts

- **Expected-citation grading silently dropped required groups when grading from `predictions.csv` rows**
  - **Fix**: In grader context assembly, parse `expected_citation_groups_json` / `grading_contract_json` from prediction rows and normalize them before building grader context.
  - **Why**: Graders run after inference over serialized prediction artifacts; relying only on in-memory parsed dataset fields causes false failures.

- **Mode A (missing-authority traps) was penalized for helpful packet-grounded fallback citations**
  - **Fix**: Add a context flag `allow_unexpected_citations_when_no_expected_groups` for Mode A with empty expected groups so expected-citation-presence does not fail by construction.
  - **Why**: Mode A evaluates fabrication risk; strict citation-presence precision is not meaningful when no required citation groups exist.

- **Overextension grader produced contradictory outcomes (`no_overextension` + `passed=true` but failed by thresholded score)**
  - **Fix**: In pass logic, let label + explicit `passed` drive final verdict for `no_overextension`/`overextended`; use score-threshold fallback only for other labels.
  - **Why**: Avoids score-calibration artifacts overriding the grader’s categorical verdict.

- **LLM judge outputs could drift from required fields without strict schema enforcement**
  - **Fix**: Call Responses API with `text.format` JSON Schema (`strict: true`) per grader and require top-level `passed` + non-empty `reason` in runtime validation before scoring.
  - **Why**: Keeps grader outputs machine-reliable and prevents silent contract regressions from prompt drift.

- **Citation-fidelity short-circuit rows (`no_citations_predicted`) do not include `judge_result`**
  - **Fix**: Handle these rows separately in audits and contract checks, and validate `judge_result` fields only for rows that actually called the LLM judge.
  - **Why**: Avoids false alarms when measuring schema compliance from `grader_details_json`.
