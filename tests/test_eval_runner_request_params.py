from pincite_evals.eval_runner import ModelConfig, _build_response_request


def test_eval_runner_drops_reasoning_for_gpt4_models():
    model_config = ModelConfig(
        name="draft",
        model="gpt-4.1",
        reasoning_effort="none",
        temperature=0.11,
        system_prompt="You are a test system prompt.",
    )
    request = _build_response_request(model_config, "Hello")

    assert request["model"] == "gpt-4.1"
    assert "reasoning" not in request
    assert request["temperature"] == 0.11


def test_eval_runner_keeps_reasoning_for_gpt5_models():
    model_config = ModelConfig(
        name="draft",
        model="gpt-5.2",
        reasoning_effort="none",
        temperature=0.22,
        system_prompt="You are a test system prompt.",
    )
    request = _build_response_request(model_config, "Hello")

    assert request["model"] == "gpt-5.2"
    assert request["reasoning"] == {"effort": "none"}
    assert request["temperature"] == 0.22

