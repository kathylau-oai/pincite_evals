# Grading

## Recommended grading stack

1. Deterministic checks
   - required headings present
   - citations present when required
   - forbidden outside citations

2. Citation existence
   - every citation references an allowed `doc_id`
   - paragraph range parses and exists in the authority text

3. Span correctness
   - for tasks with a known span trap, the cited span matches the intended paragraph range

4. Support and scope
   - the cited span supports the proposition
   - qualifiers and limits are preserved (no overextension)

5. Precedence
   - if the task includes a controlling authority, the draft must not rely on non-controlling authority as controlling
   - if a rule is limited or overruled within the packet, the limitation must be carried forward

## Calibration

- Start with deterministic and citation existence.
- Add LLM-as-judge only after you have a labeled calibration set and inter-rater checks.
