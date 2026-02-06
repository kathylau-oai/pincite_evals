# Grading

## Recommended grading stack

1. Grader 1: Deterministic citation integrity grader
   - checks that each cited `doc_id` exists in the packet
   - checks that each cited paragraph/block range is valid and exists in the source document
   - checks that the cited paragraph/block is the correct section for the claim (not just any valid block)
   - purpose: catch fake citations (hallucinated authorities) and wrong-block citations

2. Grader 2: Citation overextension grader (LLM as judge)
   - input includes:
     - the model output with citations
     - test case context
     - a natural-language note describing what overextension behavior the test case is trying to trigger
   - judge task:
     - determine whether the output overclaims beyond what the cited text supports
     - determine whether key limits/qualifiers were dropped
     - score overextension risk and provide short rationale
   - purpose: detect cases where citations exist but are stretched beyond their supported scope

3. Grader 3: Precedent/precedence grader (LLM as judge)
   - input includes:
     - the model output with citations
     - test case context
     - a natural-language note describing what precedent/precedence behavior the test case is trying to trigger
     - specific cautions (for example: controlling authority vs persuasive authority, overruled/limited authority, exceptions)
   - judge task:
     - evaluate whether the answer handles authority hierarchy and precedent constraints correctly
     - score precedence correctness and provide short rationale
   - purpose: catch legal reasoning errors even when citation formatting looks valid

## Calibration

- Run Grader 1 first, then Grader 2 and Grader 3.
- Calibrate LLM-judge prompts on a labeled set with explicit pass/fail examples for overextension and precedence.
- Track inter-rater agreement and score drift over time.
