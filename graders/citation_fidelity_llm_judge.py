import json
from typing import Any, Dict

from openai import OpenAI

from .base import GradeResult, Grader
from .llm_utils import LLMJudgeConfig, call_llm_judge, load_prompt_template


class CitationFidelityLLMJudgeGrader(Grader):
    name = "citation_fidelity_llm_judge"

    def __init__(
        self,
        *,
        model: str = "gpt-5.2",
        reasoning_effort: str = "none",
        temperature: float = 0.0,
        max_output_tokens: int = 1400,
        pass_threshold: float = 0.8,
        client: OpenAI | None = None,
    ) -> None:
        self.client = client or OpenAI()
        self.config = LLMJudgeConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.pass_threshold = pass_threshold
        self.system_prompt_template = load_prompt_template("citation_fidelity_system.md")
        self.user_prompt_template = load_prompt_template("citation_fidelity_user.md")

    def grade(self, *, prompt: str, output: str, context: Dict[str, Any]) -> GradeResult:
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

        grading_payload = {
            "task_prompt": prompt,
            "model_output": output,
            "test_case_context": context.get("test_case_context"),
            "fidelity_note": context.get("citation_fidelity_note"),
            "citation_fidelity_items": fidelity_items,
        }
        user_prompt = self.user_prompt_template.replace(
            "{{judge_payload_json}}",
            json.dumps(grading_payload, indent=2, ensure_ascii=False),
        )

        judge_result = call_llm_judge(
            client=self.client,
            config=self.config,
            system_prompt=self.system_prompt_template,
            user_prompt=user_prompt,
        )
        parsed = judge_result["parsed"]

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

        details = {
            "score": overall_score,
            "pass_threshold": threshold,
            "blocked_count": blocked_count,
            "blocked_labels": sorted(blocked_labels),
            "judge_result": parsed,
            "response_id": judge_result["response_id"],
            "response_status": judge_result["status"],
            "incomplete_reason": judge_result["incomplete_reason"],
            "usage": judge_result["usage"],
            "raw_output_text": judge_result["raw_output_text"],
        }
        return GradeResult(name=self.name, passed=passed, details=details)
