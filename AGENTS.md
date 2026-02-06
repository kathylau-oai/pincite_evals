# Pincite Evals - Agent Guidance

## Evaluation execution

- Run evaluations in parallel whenever possible to reduce end-to-end runtime.
- Support running multiple model configurations in a single evaluation run (for example: different models, reasoning effort levels, temperatures, or prompts in one pass).
- Keep run configuration explicit and reproducible so each model config can be compared apples-to-apples on the same inputs.
- Use the OpenAI Responses API (not Chat Completions) for evaluation runs.
- Use `tenacity` for retries with explicit retry policy settings (attempt limits, backoff, and retriable error types).
- Use `tqdm` progress bars for long-running evaluation and grading jobs so progress is visible.
- Use pandas DataFrames for tabular loading, transformation, and metric aggregation.
- Use Plotly for analysis and plotting outputs (especially for slice-level and comparison charts).



# API KEYs

- The repo already has OPENAI api key in the env variables, do not ask for it, run directly commands that would use it.

## Results and experiment structure

- Store all outputs under `results/` and organize each run under an experiment-specific subfolder.
- Save high-quality final artifacts for each experiment, including:
  - Final predictions and grades CSV
  - Aggregated metrics summary (quality + latency + token stats)
  - Per-model and per-slice comparison tables
- Save intermediate artifacts that enable root-cause analysis, including:
  - Raw model responses / traces
  - Judge or grader intermediate outputs
  - Item-level error annotations where available
  - Any retry/failure logs and status metadata

## Analysis quality bar

- Start with grader-centric slice analysis: compute slice-level metrics broken out by grader and compare grader behavior directly.
- Include slice-level metrics (for example by citation type, court, jurisdiction, difficulty bucket, or other relevant segments), and report each slice by grader when possible.
- Produce useful visualizations from slice-level metrics to make regressions and tradeoffs obvious.
- Include bar plots that show metrics:
  - Per grader
  - Per model run
  - Per grader x model run comparison
- Ensure plots are readable and polished (clear labels, no truncation/overlap, legible legends, sensible scales).

## Metrics requirements

- Token accounting must include reasoning tokens whenever reasoning is enabled; do not report token usage without them.
- Report token metrics with distribution stats (`avg`, `p50`, `p90`, `p95`, `p99`) and clear per-model breakdowns.
- Latency reporting must include:
  - End-to-end latency
  - Time-to-first-token (TTFT)
  - Token-by-token latency (inter-token latency) with distribution stats

## Recent learnings

- Problem -> Excerpt citation parsing accepted overly broad IDs and legacy styles -> Fix -> Enforce only `DOC_ID[P<page>.B<block>]` in parser/tests and reject legacy paragraph citations -> Why -> Keeps citations block-level, deterministic, and consistent with the current grading format.
- Problem -> Block-level packet text needed unambiguous boundaries that are easy for LLMs and parsers -> Fix -> Write annotated text with XML wrappers `<BLOCK id="DOC###.P###.B##"> ... </BLOCK>` per block -> Why -> Keeps block anchors explicit with lower token noise and simpler parsing.
- Problem -> Needed TTFT and inter-token latency metrics from Responses API calls -> Fix -> Use `client.responses.stream(...)`, timestamp `response.output_text.delta` events, and finalize with `stream.get_final_response()` for status/usage -> Why -> Captures latency distributions and token usage (including reasoning tokens) from one request path.
- Problem -> Synthetic traps could look valid when only local excerpts are checked -> Fix -> Require generation and validation to scan all packet documents for counterevidence before accepting a candidate -> Why -> Prevents ambiguous items and improves lawyer-consensus reliability.
- Problem -> Synthetic pipeline scope drifted toward all error modes -> Fix -> Explicitly scope adversarial generation to `A/C/D` and treat `B` as grader-only measurement for this phase -> Why -> Preserves focus on high-signal trap types while still tracking span errors in downstream grading.
- Problem -> Synthetic generation/verification runs were slower and configuration drifted -> Fix -> Default to `gpt-5.2` with high reasoning for generation + verifier and run mode workers in parallel with bounded concurrency -> Why -> Improves quality consistency and end-to-end throughput.
- Problem -> Deterministic conflict checks can over-reject generated items and starve mode quotas -> Fix -> Prioritize low-counterevidence targets during generation (`counterevidence_count` ascending) before LLM calls -> Why -> Maintains strict validation while keeping enough pass candidates for stable 3/3/3 selection.

