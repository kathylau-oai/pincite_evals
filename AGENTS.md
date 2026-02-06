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
- Problem -> Excerpt citation parsing accepted overly broad IDs and legacy styles -> Fix -> Enforce only `DOC_ID[P<page>.B<block>(#<hash>)?]` in parser/tests and reject legacy paragraph citations -> Why -> Keeps citations block-level, deterministic, and consistent with the current grading format.
