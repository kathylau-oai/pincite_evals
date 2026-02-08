"""
OpenAI model capability helpers.

We use the OpenAI Responses API throughout this repo. Some request parameters are
only accepted by certain model families. In particular, the `reasoning` request
block (and its nested `reasoning.effort` value) is only supported by "reasoning"
models; other models (for example `gpt-4o`, `gpt-4.1`) will 400 with:

  Unsupported parameter: 'reasoning.effort' is not supported with this model.

To keep multi-model eval runs robust, we gate those parameters based on model
name. This is intentionally a small heuristic (not a complete registry) so it
is easy to update if OpenAI model naming conventions change.
"""

def supports_reasoning_effort(model: str) -> bool:
    """
    Return True if `model` is expected to accept the Responses API `reasoning`
    parameter (including `reasoning.effort`).

    Heuristic:
    - "gpt-5*" models support reasoning controls.
    - "o*" (o-series) models support reasoning controls.

    Everything else defaults to False.
    """

    normalized_model = (model or "").strip().lower()
    if not normalized_model:
        return False

    if normalized_model.startswith("gpt-5"):
        return True

    # Covers o1, o3-mini, etc.
    if normalized_model.startswith("o"):
        return True

    return False

