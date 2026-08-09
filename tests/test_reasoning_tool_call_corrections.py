from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from onpanda import FindAndReplaceCorrectionAdapter
from onpanda.correcting_model.best_of_n_mixin import (
    test_best_of_n_judge_prompt as run_best_of_n_judge_prompt_test,
)
from onpanda.correcting_model.correcting_model import (
    CorrectingModel,
    build_test_correcting_model,
    take_policy_message,
)
from onpanda.response_templates import (
    DefaultResponseTemplate,
    flatten_messages_for_correcting,
)
from onpanda.response_templates.default import CALL_BEGIN_MARKER
from onpanda.response_templates.qwen3p5 import Qwen3p5ResponseTemplate


class ContextTokenizer:
    name_or_path = "context-tokenizer"
    texts_to_tokens = {
        "bX": [1, 2],
        "bsdm": [3, 4],
        "bs": [3],
    }
    tokens_to_text = {
        (1,): "b",
        (1, 2): "bX",
        (3,): "bs",
        (3, 4): "bsdm",
    }

    def encode(self, text, **kwargs):
        return self.texts_to_tokens[text]

    def decode(self, tokens, **kwargs):
        return self.tokens_to_text[tuple(tokens)]


def test_policy_message_normalizes_api_channels_and_finish_reason():
    policy_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "",
                    "tool_calls": None,
                },
            }
        ]
    }

    assert take_policy_message(policy_response) == {
        "role": "assistant",
        "content": "answer",
        "finish_reason": "stop",
    }

    tool_call_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
            }
        ]
    }
    assert take_policy_message(tool_call_response)["finish_reason"] == "tool_calls"


def test_reasoning_content_is_flattened_as_reasoning():
    flattened = flatten_messages_for_correcting(
        [
            {
                "role": "assistant",
                "reasoning": "",
                "reasoning_content": "thinking",
                "content": "answer",
                "finish_reason": "stop",
            }
        ],
        DefaultResponseTemplate(),
    )

    assert flattened[0]["content"].startswith("<|reasoning|>thinking")
    assert "reasoning_content" not in flattened[0]


def test_correcting_call_preserves_special_tokens():
    class Adapter:
        def build_correction_prompt(self, messages):
            return messages

        def apply(self, messages, far_text, tools=None, adapter_policy=None):
            return {"correction": {}, "partial_messages": messages}

    def chat_correcting(messages, **kwargs):
        assert kwargs == {
            "return_dict": True,
            "skip_special_tokens": False,
            "tools": ["tool"],
            "tool_choice": "none",
        }
        return {
            "model": "correcting",
            "choices": [{"message": {"content": "far"}}],
        }

    chat_correcting.model = "correcting"
    CorrectingModel(chat_correcting, Adapter()).correct(
        [{"role": "user", "content": "q"}], tools=["tool"]
    )


def test_best_of_n_judge_accepts_null_tool_calls():
    assert run_best_of_n_judge_prompt_test(build_test_correcting_model()) == 2


@pytest.mark.parametrize(
    "template", [DefaultResponseTemplate(), Qwen3p5ResponseTemplate()]
)
def test_length_finish_reason_is_not_coerced_to_tool_calls(template):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"index": 0, "function": {"name": "read", "arguments": '{"path":'}}
        ],
    }
    text = template.apply(message)["templated_prompt"]

    assert template.parse(text, finish_reason="length")["finish_reason"] == "length"


def test_qwen_template_round_trips_open_tool_call_states():
    template = Qwen3p5ResponseTemplate()

    empty_channel = {"role": "assistant", "content": "", "tool_calls": []}
    assert (
        template.parse(template.apply(empty_channel)["templated_prompt"])["tool_calls"]
        == []
    )

    open_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"index": 0, "type": "function"}],
    }
    parsed = template.parse(template.apply(open_call)["templated_prompt"])
    assert parsed["tool_calls"][0]["function"]["name"] == ""


def test_default_template_parses_partial_tool_calls_boundary():
    template = DefaultResponseTemplate()
    assert template.parse("answer<|tool_calls|>") == {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [],
    }
    assert template.parse("answer<|tool_calls|>\n") == {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [],
    }
    for marker_prefix in ("<", "<<", "<|to", "<|tool_call", "<|tool_calls|"):
        assert template.parse("answer" + marker_prefix) == {
            "role": "assistant",
            "content": "answer" + marker_prefix,
        }


def test_default_template_requires_complete_call_begin_marker():
    template = DefaultResponseTemplate()
    tool_calls_begin = template.tool_calls_begin_marker
    assert template.parse(tool_calls_begin + CALL_BEGIN_MARKER) == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"index": 0}],
    }
    for marker_prefix in (
        "\n",
        "x",
        "<",
        CALL_BEGIN_MARKER[:3],
        CALL_BEGIN_MARKER[:-1],
    ):
        assert template.parse(tool_calls_begin + marker_prefix) == {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
        }


def test_qwen_template_preserves_open_reasoning_trailing_newline():
    template = Qwen3p5ResponseTemplate()
    message = {"role": "assistant", "reasoning": "think\n"}
    text = template.apply(message)["templated_prompt"]

    assert template.parse(text) == message


def test_policy_prefix_is_a_prefix_of_the_corrected_token_sequence():
    adapter = FindAndReplaceCorrectionAdapter(
        tokenizer=ContextTokenizer(), max_replacement_tokens=1
    )

    partial = adapter.build_partial_templated_prompt(
        {"role": "assistant", "content": "bX"},
        {"role": "assistant", "content": "bsdm"},
    )

    assert partial == {"templated_prompt": "bs", "replacement": "bs"}


def test_complete_early_stop_is_not_rejected_as_no_op():
    correcting_adapter = FindAndReplaceCorrectionAdapter(max_replacement_tokens=20)
    policy_adapter = FindAndReplaceCorrectionAdapter(
        max_replacement_tokens=20,
        response_template={"name_or_path": "Qwen/Qwen3.6-35B-A3B"},
    )
    split = correcting_adapter.special_tokens["split"]
    stop = correcting_adapter.special_tokens["stop"]
    result = correcting_adapter.apply(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "abc", "finish_reason": "stop"},
        ],
        split + "bc" + split + "0" + split + stop + split,
        adapter_policy=policy_adapter,
    )

    assert result["partial_messages"][-1] == {
        "role": "assistant",
        "content": "a",
        "finish_reason": "stop",
    }

    result = correcting_adapter.apply(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "abc", "finish_reason": "length"},
        ],
        split + "bc" + split + "0" + split + "bc" + stop + split,
        adapter_policy=policy_adapter,
    )
    assert result["partial_messages"][-1]["finish_reason"] == "stop"


def test_complete_replacement_obeys_policy_token_limit():
    adapter = FindAndReplaceCorrectionAdapter(max_replacement_tokens=1)
    correcting_model = CorrectingModel(
        SimpleNamespace(model="correcting"), adapter, max_correction_attempts=1
    )
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "abc", "finish_reason": "stop"},
    ]

    def correct(self, messages, tools=None, adapter_policy=None):
        return dict(
            correction=dict(
                messages_location=dict(path_keys=[1, "content"], char_index=1)
            ),
            partial_messages=[
                messages[0],
                {"role": "assistant", "content": "axyz", "finish_reason": "stop"},
            ],
            correction_response={},
        )

    correcting_model.correct = MethodType(correct, correcting_model)
    policy_prefixes = []

    def policy(policy_messages, **kwargs):
        prefix = policy_messages[-1]["content"]
        policy_prefixes.append(prefix)
        return dict(
            choices=[
                dict(
                    message=dict(role="assistant", content=prefix + " generated"),
                    finish_reason="stop",
                )
            ]
        )

    policy.model = "policy"
    result = correcting_model.correct_and_rollout(
        messages, policy, adapter_policy=adapter
    )
    assert policy_prefixes == ["ax"]
    assert result["correction"]["continue_prefix_right40"] == "ax"
    assert result["corrected_messages"][-1]["content"] == "ax generated"

    adapter.max_replacement_tokens = 20
    result = correcting_model.correct_and_rollout(
        messages, policy, adapter_policy=adapter
    )
    assert policy_prefixes == ["ax"]
    assert "continue_prefix_right40" not in result["correction"]
    assert result["corrected_messages"][-1] == {
        "role": "assistant",
        "content": "axyz",
        "finish_reason": "stop",
    }


def test_complete_no_op_is_rejected():
    adapter = FindAndReplaceCorrectionAdapter(max_replacement_tokens=20)
    split = adapter.special_tokens["split"]
    stop = adapter.special_tokens["stop"]
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "abc", "finish_reason": "stop"},
    ]
    result = adapter.apply(
        messages,
        split + "bc" + split + "0" + split + "bc" + stop + split,
    )

    assert result["correction"]["messages_location"]["not_found"] is True
    assert result["partial_messages"] == messages


@pytest.mark.parametrize(
    "response_template",
    [None, {"name_or_path": "Qwen/Qwen3.6-35B-A3B"}],
)
def test_structured_tool_calls_survive_policy_continuation(response_template):
    adapter = FindAndReplaceCorrectionAdapter(
        max_replacement_tokens=200, response_template=response_template
    )
    correcting_model = CorrectingModel(
        SimpleNamespace(model="correcting"), adapter, max_correction_attempts=1
    )

    def correct(self, messages, tools=None, adapter_policy=None):
        return {
            "correction": {
                "messages_location": {"path_keys": [1, "content"], "char_index": 0}
            },
            "partial_messages": [
                messages[0],
                {
                    "role": "assistant",
                    "content": "fixed",
                    "tool_calls": [
                        {
                            "index": 0,
                            "type": "function",
                            "id": "prefix_call",
                            "function": {
                                "name": "read",
                                "arguments": '{"path": "/tmp/fixed"}',
                            },
                        }
                    ],
                },
            ],
            "correction_response": {},
        }

    correcting_model.correct = MethodType(correct, correcting_model)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]

    class Policy:
        model = "policy"

        def __call__(self, messages, **kwargs):
            assert kwargs["skip_special_tokens"] is False
            assert kwargs["tool_choice"] == "none"
            assert kwargs["tools"] == tools
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "type": "function",
                                    "id": "call_0",
                                    "function": {
                                        "name": "read",
                                        "arguments": '{"path": "/tmp/a"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }

    corrected = correcting_model.correct_and_rollout(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "fixed",
                "tool_calls": [
                    {
                        "index": 0,
                        "type": "function",
                        "id": "prefix_call",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "/tmp/bad"}',
                        },
                    }
                ],
                "finish_reason": "tool_calls",
            },
        ],
        Policy(),
        adapter_policy=adapter,
        tools=tools,
    )["corrected_messages"][-1]

    assert corrected["content"] == "fixed"
    assert [call["function"] for call in corrected["tool_calls"]] == [
        {"name": "read", "arguments": '{"path": "/tmp/fixed"}'},
        {"name": "read", "arguments": '{"path": "/tmp/a"}'},
    ]
    assert corrected["finish_reason"] == "tool_calls"


def test_mislabeled_content_continuation_does_not_leak_think_end():
    adapter = FindAndReplaceCorrectionAdapter(
        max_replacement_tokens=200,
        response_template={"name_or_path": "Qwen/Qwen3.6-35B-A3B"},
    )
    correcting_model = CorrectingModel(
        SimpleNamespace(model="correcting"), adapter, max_correction_attempts=1
    )
    rejected_message = {
        "role": "assistant",
        "reasoning": "think",
        "content": "answer /tmp/b",
        "finish_reason": "stop",
    }
    partial_message = {
        "role": "assistant",
        "reasoning": "think",
        "content": "answer /tmp/a",
    }
    prefix = adapter.build_partial_templated_prompt(
        rejected_message, partial_message
    )["templated_prompt"]
    correcting_model.correct = Mock(
        return_value={
            "correction": {
                "messages_location": {
                    "path_keys": [1, "content"],
                    "char_index": 0,
                }
            },
            "partial_messages": [{"role": "user", "content": "q"}, partial_message],
        }
    )
    policy = Mock(
        return_value={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        # mxlm echoes the prefix before returning the vLLM response.
                        "content": (
                            prefix
                            + "<tool_call>\n"
                            "<function=read_file>\n"
                            "<parameter=path>\n/tmp/a.txt\n</parameter>\n"
                            "</function>\n"
                            "</tool_call>"
                        ),
                        "reasoning": ".txt\n",
                        "reasoning_content": ".txt\n",
                        "tool_calls": [],
                    },
                }
            ]
        }
    )
    policy.model = "policy"
    corrected = correcting_model.correct_and_rollout(
        [{"role": "user", "content": "q"}, rejected_message],
        policy,
        adapter_policy=adapter,
    )["corrected_messages"][-1]

    assert corrected["reasoning"] == "think"
    assert corrected["content"] == "answer /tmp/a.txt"
    assert "</think>" not in corrected["content"]
    assert corrected["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path": "/tmp/a.txt"}',
    }
    assert corrected["finish_reason"] == "tool_calls"
