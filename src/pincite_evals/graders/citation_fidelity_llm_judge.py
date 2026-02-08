from typing import Any, Dict

from .base import GradeResult
from .llm_judge_base import BaseLLMJudgeGrader


class CitationFidelityLLMJudgeGrader(BaseLLMJudgeGrader):
    name = "citation_fidelity_llm_judge"
    system_prompt_file = "citation_fidelity_system.txt"
    user_prompt_file = "citation_fidelity_user.txt"

    def _validate_context(self, context: Dict[str, Any]) -> GradeResult | None:
        fidelity_items = context.get("citation_fidelity_items", [])
        if not fidelity_items:
            return GradeResult(
                name=self.name,
                passed=False,
                details={
                    "error": "Missing required context key: citation_fidelity_items",
                    "score": 0.0,
                },
            )
        return None

    def _build_grading_payload(self, *, prompt: str, output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_prompt": prompt,
            "model_output": output,
            "test_case_context": context.get("test_case_context"),
            "fidelity_note": context.get("citation_fidelity_note"),
            "citation_fidelity_items": context.get("citation_fidelity_items", []),
        }

    def _response_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "passed": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["passed", "reason"],
        }

    def _compute_grade(self, *, parsed: Dict[str, Any], context: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        judge_passed = bool(parsed.get("passed", False))

        # Deterministic guard: if any cited token does not exist in the packet, it is hallucinated.
        hallucinated_tokens = [
            str(item.get("citation_token", ""))
            for item in context.get("citation_fidelity_items", [])
            if isinstance(item, dict) and item.get("exists_in_packet") is False
        ]
        passed = judge_passed and (len(hallucinated_tokens) == 0)

        score = 1.0 if passed else 0.0
        return passed, {"score": score, "hallucinated_tokens": hallucinated_tokens}
