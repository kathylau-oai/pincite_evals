from typing import Any, Dict

from .llm_judge_base import BaseLLMJudgeGrader


class CitationOverextensionLLMJudgeGrader(BaseLLMJudgeGrader):
    name = "citation_overextension_llm_judge"
    system_prompt_file = "citation_overextension_system.txt"
    user_prompt_file = "citation_overextension_user.txt"

    def _build_grading_payload(self, *, prompt: str, output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_prompt": prompt,
            "model_output": output,
            "test_case_context": context.get("test_case_context"),
            "overextension_trigger_note": context.get("overextension_trigger_note"),
            "overextension_cautions": context.get("overextension_cautions"),
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
        score = 1.0 if judge_passed else 0.0
        return judge_passed, {"score": score}
