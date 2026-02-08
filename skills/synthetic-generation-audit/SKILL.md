---
name: synthetic-generation-audit
description: Run full synthetic data generation for all packets, audit ACCEPTED/REJECTED datapoints and request traces, and produce prompt-only recommendation output (no implementation) with examples.
---

# Synthetic Generation Audit

## Overview
Run the repo synthetic generation pipeline end-to-end, then produce a clear quality audit across accepted and rejected datapoints, trace health, user-query realism, and prompt-only improvement recommendations.

This skill is designed to replicate the same workflow consistently and quickly.

## What this skill must do
1. Trigger generation using the repo script:
```bash
bash src/pincite_evals/synthetic_generation/run_all_packets.sh <run_timestamp>
```
2. Audit ACCEPTED and REJECTED datapoints across all packets.
3. Explain rejected error modes in clear language with concrete examples.
4. Review request traces (generation + validation) to confirm requests worked properly.
5. Review accepted user queries for realism and whether they match expected legal-assistant usage.
6. Output prompt modification recommendations only.
7. **Do not implement recommendations unless explicitly asked.**

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
- `results/synthetic_generation_audit/<run_timestamp>/validation_all.csv`
- `results/synthetic_generation_audit/<run_timestamp>/accepted_items.csv`
- `results/synthetic_generation_audit/<run_timestamp>/rejected_items.csv`
- `results/synthetic_generation_audit/<run_timestamp>/trace_health.csv`
- `results/synthetic_generation_audit/<run_timestamp>/prompt_recommendations.csv`
- `results/synthetic_generation_audit/<run_timestamp>/summary_metrics.csv`

## Interpretation checklist
When presenting results, include:
1. Run ID and packet coverage.
2. Accepted vs rejected totals (overall and by packet + mode).
3. Rejected mode breakdown and dominant failure reasons.
4. Whether any accepted items look questionable.
5. Trace health summary (completion/error/incomplete/retries) with a few concrete trace examples.
6. Accepted query realism summary with examples.
7. Prompt-only recommendations prioritized by impact.

## Guardrails
- Use pandas DataFrames for CSV manipulation.
- Keep explanations in plain English with specific examples.
- Recommendation section must be explicit that it is **not implemented**.
- If a run is missing packet folders, fail clearly with actionable guidance.
- Ensure generated test cases are realistic; do not over-engineer the `user_query` to trigger the error mode at the expense of sounding like a real lawyer request.
