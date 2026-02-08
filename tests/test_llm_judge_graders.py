import json

from typing import cast

from openai import OpenAI

from pincite_evals.graders.citation_fidelity_llm_judge import (
    CitationFidelityLLMJudgeGrader,
)
from pincite_evals.graders.citation_overextension_llm_judge import (
    CitationOverextensionLLMJudgeGrader,
)
from pincite_evals.graders.precedence_llm_judge import PrecedenceLLMJudgeGrader


class FakeUsage:
    def model_dump(self, mode: str = "json"):
        return {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


class FakeIncompleteDetails:
    def __init__(self, reason: str):
        self.reason = reason


class FakeResponse:
    def __init__(
        self,
        *,
        output_text: str,
        status: str = "completed",
        incomplete_reason: str | None = None,
    ):
        self.id = "resp_test_123"
        self.status = status
        self.output_text = output_text
        self.usage = FakeUsage()
        self.incomplete_details = None
        if incomplete_reason is not None:
            self.incomplete_details = FakeIncompleteDetails(incomplete_reason)


class FakeResponsesAPI:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: FakeResponse):
        self.responses = FakeResponsesAPI(response)


def test_citation_fidelity_llm_judge_fails_hallucinated_label():
    response_payload = {
        "overall_score": 0.92,
        "passed": True,
        "reason": "At least one citation is hallucinated.",
        "summary": "One citation was hallucinated.",
        "item_results": [
            {
                "citation_token": "DOC1[P001.B01#AAAA]",
                "label": "hallucinated",
                "score": 0.0,
                "passed": False,
                "reason": "Block does not support claim.",
                "evidence_excerpt": "",
            }
        ],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = CitationFidelityLLMJudgeGrader(client=cast(OpenAI, fake_client))

    result = grader.grade(
        prompt="p",
        output="o",
        context={
            "citation_fidelity_items": [
                {"citation_token": "DOC1[P001.B01#AAAA]", "exists_in_packet": False}
            ]
        },
    )
    assert result.passed is False
    assert result.details["hallucinated_tokens"] == ["DOC1[P001.B01#AAAA]"]
    last_request = fake_client.responses.last_request
    assert last_request is not None
    assert last_request["model"] == "gpt-5.2"
    assert last_request["service_tier"] == "priority"
    assert last_request["reasoning"] == {"effort": "none"}
    assert last_request["text"]["format"]["type"] == "json_schema"
    assert "temperature" in last_request


def test_overextension_llm_judge_passes_clean_label():
    response_payload = {
        "label": "no_overextension",
        "score": 0.93,
        "passed": True,
        "reason": "Claims remained within citation bounds.",
        "evidence": ["Short quote"],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = CitationOverextensionLLMJudgeGrader(client=cast(OpenAI, fake_client))

    result = grader.grade(
        prompt="p", output="o", context={"overextension_trigger_note": "trap note"}
    )
    assert result.passed is True
    assert result.details["score"] == 1.0


def test_overextension_llm_judge_no_overextension_label_overrides_low_score():
    response_payload = {
        "label": "no_overextension",
        "score": 0.12,
        "passed": True,
        "reason": "Claims stayed within source qualifiers.",
        "evidence": ["Short quote"],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = CitationOverextensionLLMJudgeGrader(client=cast(OpenAI, fake_client))

    result = grader.grade(
        prompt="p", output="o", context={"overextension_trigger_note": "trap note"}
    )
    assert result.passed is True
    assert result.details["score"] == 1.0


def test_precedence_llm_judge_fails_precedence_error():
    response_payload = {
        "label": "precedence_error",
        "score": 0.25,
        "passed": False,
        "reason": "Non-controlling authority treated as controlling.",
        "evidence": ["Short quote"],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = PrecedenceLLMJudgeGrader(client=cast(OpenAI, fake_client))

    result = grader.grade(
        prompt="p", output="o", context={"precedence_trigger_note": "note"}
    )
    assert result.passed is False
    assert result.details["score"] == 0.0


def test_llm_judge_requires_non_empty_reason():
    response_payload = {
        "label": "no_overextension",
        "score": 0.95,
        "passed": True,
        "reason": "",
        "evidence": ["Short quote"],
    }
    fake_client = FakeOpenAIClient(
        FakeResponse(output_text=json.dumps(response_payload))
    )
    grader = CitationOverextensionLLMJudgeGrader(client=cast(OpenAI, fake_client))

    try:
        grader.grade(
            prompt="p", output="o", context={"overextension_trigger_note": "trap note"}
        )
    except ValueError as error:
        assert "reason" in str(error)
        return
    raise AssertionError("Expected a ValueError when required 'reason' is empty.")


def test_llm_judge_schemas_require_passed_and_reason():
    graders = [
        CitationFidelityLLMJudgeGrader(
            client=cast(OpenAI, FakeOpenAIClient(FakeResponse(output_text="{}")))
        ),
        CitationOverextensionLLMJudgeGrader(
            client=cast(OpenAI, FakeOpenAIClient(FakeResponse(output_text="{}")))
        ),
        PrecedenceLLMJudgeGrader(
            client=cast(OpenAI, FakeOpenAIClient(FakeResponse(output_text="{}")))
        ),
    ]

    for grader in graders:
        schema = grader._response_schema()
        required = schema["required"]
        assert "passed" in required
        assert "reason" in required


def test_llm_judge_drops_reasoning_for_gpt4_models():
    response_payload = {
        "passed": True,
        "reason": "Looks good.",
    }
    fake_client = FakeOpenAIClient(FakeResponse(output_text=json.dumps(response_payload)))
    grader = CitationOverextensionLLMJudgeGrader(
        client=cast(OpenAI, fake_client),
        model="gpt-4.1",
        reasoning_effort="high",  # should be ignored/dropped for this model family
        temperature=0.42,
    )

    grader.grade(prompt="p", output="o", context={})

    last_request = fake_client.responses.last_request
    assert last_request is not None
    assert last_request["model"] == "gpt-4.1"
    assert "reasoning" not in last_request
    assert last_request["temperature"] == 0.42
