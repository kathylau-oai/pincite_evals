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
  - **Fix**: Enforce only `DOC_ID[P<page>.B<block>]` in parser/tests and reject legacy paragraph citations.
  - **Why**: Keeps citations block-level, deterministic, and consistent with the current grading format.

- **Packet text needed unambiguous, low-noise block boundaries**
  - **Fix**: Render annotated packet text with XML wrappers: `<BLOCK id="DOC###.P###.B##"> ... </BLOCK>` per block.
  - **Why**: Makes block anchors explicit with lower token noise and simpler parsing.

- **Packet annotations exposed XML block IDs while generation expected bracket citations**
  - **Fix**: Accept both `DOC###.P###.B##` and `DOC###[P###.B##]` and normalize to canonical tokens before schema/deterministic checks.
  - **Why**: Keeps generation/validation compatible with updated packet rendering without breaking downstream grading.

- **Verifier false-rejected due to mixed citation notation between item JSON and packet corpus**
  - **Fix**: Render verifier item payload citations in dotted block-id format and explicitly instruct the verifier that `DOC001[P001.B01]` and `DOC001.P001.B01` are equivalent.
  - **Why**: Prevents notation-only rejections and keeps validation focused on substantive grading quality.

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

- **Synthetic item CSV prompt field can vary between `prompt` and `user_query`**
  - **Fix**: In eval loaders, accept both names and normalize to the active prompt column before validation.
  - **Why**: Prevents brittle runner failures when packet datasets come from different pipeline revisions.

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
