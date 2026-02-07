from typing import Any, Dict

from .llm_judge_base import BaseLLMJudgeGrader


class PrecedenceLLMJudgeGrader(BaseLLMJudgeGrader):
    name = "precedence_llm_judge"
    system_prompt_file = "precedence_system.txt"
    user_prompt_file = "precedence_user.txt"

    def _build_grading_payload(self, *, prompt: str, output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_prompt": prompt,
            "model_output": output,
            "test_case_context": context.get("test_case_context"),
            "precedence_trigger_note": context.get("precedence_trigger_note"),
            "precedence_cautions": context.get("precedence_cautions"),
            "authority_graph": context.get("authority_graph"),
        }

    def _compute_grade(self, *, parsed: Dict[str, Any], context: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        score = float(parsed.get("score", 0.0))
        threshold = float(context.get("pass_threshold", self.pass_threshold))
        label = str(parsed.get("label", "")).strip().lower()
        model_passed = bool(parsed.get("passed", False))

        passed = model_passed and (score >= threshold) and (label != "precedence_error")

        return passed, {
            "score": score,
            "label": label,
            "pass_threshold": threshold,
        }
