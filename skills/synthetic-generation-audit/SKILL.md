---
name: synthetic-generation-audit
description: Run full synthetic data generation for all packets, build parsed evidence tables for accepted/rejected datapoints and traces, then perform model-led root-cause analysis and prompt-only recommendations (no implementation).
---

# Synthetic Generation Audit

## Overview
Run the repo synthetic generation pipeline end-to-end, then prepare evidence tables that make LLM/manual reasoning easier.

Critical principle: scripts do parsing/formatting; the model does the reasoning.

## Reasoning policy (hard rule)
- Do **not** treat scripted fields (`llm_risk_flags_json`, `llm_reason`, `final_rejection_reason`) as ground truth.
- Use those fields only as clues.
- Determine root causes and error modes by reading the underlying datapoint evidence itself:
  - `user_query`
  - `scenario_facts_json`
  - `expected_citation_groups_json`
  - verifier outputs (`llm_verdict`, `llm_reason`, `validation_request_status`)
  - deterministic fields + trace evidence
- Recommendations must come from this model-led reasoning and must be marked as recommendation-only (not implemented).

## What this skill must do
1. Trigger generation using the repo script:
```bash
bash src/pincite_evals/synthetic_generation/run_all_packets.sh <run_timestamp>
```
2. Build parsed evidence tables for ACCEPTED/REJECTED datapoints and traces.
3. Perform root-cause/error-mode analysis using evidence tables.
4. Explain rejected patterns in clear language with concrete examples.
5. Review traces to confirm request health and distinguish quality issues from infra/runtime issues.
6. Review accepted queries for realism and likely production behavior.
7. Focus prompt suggestions on error-mode performance by reviewing test case data and verifier reasoning together:
   - test case fields: `user_query`, `scenario_facts_json`, `expected_citation_groups_json`, `target_error_mode`
   - verifier fields: `llm_verdict`, `llm_reason`, `validation_request_status`, `final_rejection_reason`
   - for each mode (`A`, `C`, `D`), identify repeat failure patterns and map each pattern to a concrete prompt-text change suggestion
8. Output prompt modification recommendations only.
9. **Do not implement recommendations unless explicitly asked.**

## Error-mode prompt-iteration focus (required)
Use this loop when analyzing and proposing prompt changes:
1. Slice rejected and accepted datapoints by `target_error_mode`.
2. For each mode, review at least several concrete examples from test case payload fields (`user_query`, `scenario_facts_json`, `expected_citation_groups_json`).
3. Cross-check each example with verifier outcome/reason (`llm_verdict`, `llm_reason`, `validation_request_status`, `final_rejection_reason`).
4. Decide whether the failure is mainly:
   - prompt clarity/coverage issue,
   - grading-contract specificity issue,
   - infra/runtime issue (for example rate limit/retry failure).
5. Propose prompt-only edits that directly target the mode-specific failure pattern, with one example before/after prompt instruction when possible.

## Repeatable command
Preferred wrapper command:
```bash
bash skills/synthetic-generation-audit/scripts/run_and_analyze.sh
```

Optional:
- Reuse an existing run without re-generating:
```bash
bash skills/synthetic-generation-audit/scripts/run_and_analyze.sh <run_timestamp> --skip-generation
```

## Outputs
The script writes outputs under:
- `results/synthetic_generation_audit/<run_timestamp>/analysis_summary.md`
- `results/synthetic_generation_audit/<run_timestamp>/summary_metrics.csv`
- `results/synthetic_generation_audit/<run_timestamp>/validation_all.csv`
- `results/synthetic_generation_audit/<run_timestamp>/accepted_items.csv`
- `results/synthetic_generation_audit/<run_timestamp>/rejected_items.csv`
- `results/synthetic_generation_audit/<run_timestamp>/trace_health.csv`
- `results/synthetic_generation_audit/<run_timestamp>/rejected_reasoning_evidence.csv`
- `results/synthetic_generation_audit/<run_timestamp>/accepted_reasoning_evidence.csv`
- `results/synthetic_generation_audit/<run_timestamp>/trace_reasoning_evidence.csv`
- `results/synthetic_generation_audit/<run_timestamp>/analyst_workflow.md`

## Interpretation checklist
When presenting results, include:
1. Run ID and packet coverage.
2. Accepted vs rejected totals (overall and by packet + mode).
3. Root-cause clusters you inferred from evidence (not from pre-labeled flags alone).
4. Whether any accepted items should likely have been rejected.
5. Trace health summary (completion/error/incomplete/retries) with examples.
6. Accepted query realism summary with examples.
7. Error-mode findings (`A`/`C`/`D`) that tie test case evidence and verifier reasons to prompt suggestions.
8. Prompt-only recommendations prioritized by impact, explicitly marked not implemented.

## Guardrails
- Use pandas DataFrames for CSV manipulation.
- Keep explanations in plain English with specific examples.
- Recommendation section must be explicit that it is **not implemented**.
- If a run is missing packet folders, fail clearly with actionable guidance.
- Ensure generated test cases are realistic; do not over-engineer the `user_query` to trigger the error mode at the expense of sounding like a real lawyer request.
