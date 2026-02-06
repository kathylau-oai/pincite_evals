import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class LLMJudgeConfig:
    model: str = "gpt-5.2"
    reasoning_effort: str = "none"
    temperature: float = 0.0
    max_output_tokens: int = 1400


def load_prompt_template(file_name: str) -> str:
    prompt_path = PROMPTS_DIR / file_name
    return prompt_path.read_text(encoding="utf-8").strip()


def parse_json_object(output_text: str) -> Dict[str, Any]:
    stripped_text = output_text.strip()
    if not stripped_text:
        raise ValueError("LLM output was empty; expected a JSON object.")

    try:
        parsed = json.loads(stripped_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped_text, re.DOTALL)
        if match is None:
            raise ValueError("Could not find JSON object in LLM output.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON was not an object.")
    return parsed


def call_llm_judge(
    *,
    client: OpenAI,
    config: LLMJudgeConfig,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "model": config.model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {"effort": config.reasoning_effort},
        "max_output_tokens": config.max_output_tokens,
    }
    if config.reasoning_effort == "none":
        request["temperature"] = config.temperature

    response = client.responses.create(**request)
    output_text = response.output_text or ""

    incomplete_reason: Optional[str] = None
    if response.incomplete_details is not None:
        incomplete_reason = response.incomplete_details.reason
    if response.status == "incomplete" and incomplete_reason == "max_output_tokens" and not output_text.strip():
        raise RuntimeError(
            "LLM response was incomplete due to max_output_tokens and had empty output_text."
        )

    parsed = parse_json_object(output_text)

    usage = None
    if response.usage is not None:
        usage = response.usage.model_dump(mode="json")

    return {
        "response_id": response.id,
        "status": response.status,
        "incomplete_reason": incomplete_reason,
        "usage": usage,
        "raw_output_text": output_text,
        "parsed": parsed,
        "request": request,
    }
