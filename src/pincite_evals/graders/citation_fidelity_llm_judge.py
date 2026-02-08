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
                "overall_score": {"type": "number"},
                "passed": {"type": "boolean"},
                "reason": {"type": "string"},
                "summary": {"type": "string"},
                "item_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "citation_token": {"type": "string"},
                            "label": {
                                "type": "string",
                                "enum": [
                                    "accurate",
                                    "wrong_block_like",
                                    "hallucinated",
                                    "metadata_mismatch",
                                    "unsupported",
                                ],
                            },
                            "score": {"type": "number"},
                            "passed": {"type": "boolean"},
                            "reason": {"type": "string"},
                            "evidence_excerpt": {"type": "string"},
                        },
                        "required": [
                            "citation_token",
                            "label",
                            "score",
                            "passed",
                            "reason",
                            "evidence_excerpt",
                        ],
                    },
                },
            },
            "required": ["overall_score", "passed", "reason", "summary", "item_results"],
        }

    def _compute_grade(self, *, parsed: Dict[str, Any], context: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        overall_score = float(parsed.get("overall_score", 0.0))
        threshold = float(context.get("pass_threshold", self.pass_threshold))
        blocked_labels = {"hallucinated", "wrong_block_like", "unsupported"}

        blocked_count = 0
        for item_result in parsed.get("item_results", []):
            item_label = item_result.get("label")
            if item_label in blocked_labels:
                blocked_count += 1

        model_passed = bool(parsed.get("passed", False))
        passed = model_passed and (overall_score >= threshold) and (blocked_count == 0)

        return passed, {
            "score": overall_score,
            "pass_threshold": threshold,
            "blocked_count": blocked_count,
            "blocked_labels": sorted(blocked_labels),
        }
