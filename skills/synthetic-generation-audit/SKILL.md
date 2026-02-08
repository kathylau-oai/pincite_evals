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
7. Output prompt modification recommendations only.
8. **Do not implement recommendations unless explicitly asked.**

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
7. Prompt-only recommendations prioritized by impact, explicitly marked not implemented.

## Guardrails
- Use pandas DataFrames for CSV manipulation.
- Keep explanations in plain English with specific examples.
- Recommendation section must be explicit that it is **not implemented**.
- If a run is missing packet folders, fail clearly with actionable guidance.
- Ensure generated test cases are realistic; do not over-engineer the `user_query` to trigger the error mode at the expense of sounding like a real lawyer request.
