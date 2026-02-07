import json
from typing import Any, Dict

from openai import OpenAI

from .base import GradeResult, Grader
from .llm_utils import LLMJudgeConfig, call_llm_judge, load_prompt_template, render_prompt_template


class BaseLLMJudgeGrader(Grader):
    system_prompt_file: str
    user_prompt_file: str

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        reasoning_effort: str = "none",
        temperature: float = 0.0,
        pass_threshold: float = 0.8,
        client: OpenAI | None = None,
    ) -> None:
        self.client = client or OpenAI()
        self.config = LLMJudgeConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        self.pass_threshold = pass_threshold
        self.system_prompt_template = load_prompt_template(self.system_prompt_file)
        self.user_prompt_template = load_prompt_template(self.user_prompt_file)

    def _validate_context(self, context: Dict[str, Any]) -> GradeResult | None:
        return None

    def _build_grading_payload(self, *, prompt: str, output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _compute_grade(self, *, parsed: Dict[str, Any], context: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        raise NotImplementedError

    def grade(self, *, prompt: str, output: str, context: Dict[str, Any]) -> GradeResult:
        context_validation = self._validate_context(context)
        if context_validation is not None:
            return context_validation

        grading_payload = self._build_grading_payload(prompt=prompt, output=output, context=context)
        user_prompt = render_prompt_template(
            self.user_prompt_template,
            {"judge_payload_json": json.dumps(grading_payload, indent=2, ensure_ascii=False)},
        )

        judge_result = call_llm_judge(
            client=self.client,
            config=self.config,
            system_prompt=self.system_prompt_template,
            user_prompt=user_prompt,
        )
        parsed = judge_result["parsed"]

        passed, mode_details = self._compute_grade(parsed=parsed, context=context)
        details = {
            **mode_details,
            "judge_result": parsed,
            "response_id": judge_result["response_id"],
            "response_status": judge_result["status"],
            "incomplete_reason": judge_result["incomplete_reason"],
            "usage": judge_result["usage"],
            "raw_output_text": judge_result["raw_output_text"],
        }
        return GradeResult(name=self.name, passed=passed, details=details)
