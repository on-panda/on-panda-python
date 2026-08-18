import json
from copy import deepcopy
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from onpanda import FindAndReplaceCorrectionAdapter
from onpanda import utf8_tokenizer
from onpanda.correcting_model.best_of_n_mixin import (
    test_best_of_n_judge_prompt as run_best_of_n_judge_prompt_test,
)
from onpanda.correcting_model.correcting_model import (
    CorrectingModel,
    build_test_correcting_model,
    take_policy_message,
)
from onpanda.parser import PandaTree
from onpanda.response_templates import (
    DefaultResponseTemplate,
    build_messages_location,
    build_templated_char_index,
    flatten_messages_for_correcting,
)
from onpanda.response_templates.default import CALL_BEGIN_MARKER
from onpanda.response_templates.qwen3p5 import (
    FUNCTION_BEGIN,
    PARAMETER_BEGIN,
    Qwen3p5ResponseTemplate,
    THINK_END,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
)
from onpanda.response_templates.step3p5 import Step3p5ResponseTemplate
from onpanda.response_templates.partial_json import parse_partial_json_object
from onpanda.test_utils import (
    ERROR_TYPES,
    get_test_trajectories,
)


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


def test_default_far_refs_build_expected_partials():
    trajectories = get_test_trajectories()
    assert tuple(key.removeprefix("error_type:") for key in trajectories) == ERROR_TYPES
    assert (
        get_test_trajectories("redundant_call")
        == trajectories["error_type:redundant_call"]
    )
    for key, trajectory in trajectories.items():
        error_type = key.removeprefix("error_type:")
        messages = trajectory["rejected_messages"]
        tools = trajectory.get("tools")
        partial_message = trajectory["partial_messages"][
            trajectory["fork_message_index"]
        ]
        for template in (Qwen3p5ResponseTemplate(), Step3p5ResponseTemplate()):
            text = template.apply(partial_message)["templated_prompt"]
            parsed_message = template.parse(
                text,
                messages=messages[: trajectory["fork_message_index"]],
                tools=tools,
                finish_reason=partial_message.get("finish_reason"),
            )
            assert template.apply(parsed_message)["templated_prompt"] == text


@pytest.mark.parametrize(
    "template", [Qwen3p5ResponseTemplate(), Step3p5ResponseTemplate()]
)
def test_structured_tool_call_locations_map_each_channel(template):
    arguments = '{"path": "/tmp/a.txt", "limit": 1}'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": arguments}}],
    }
    applied = dict(template.apply(message), message_index=1, message=message)
    prompt = applied["templated_prompt"]
    arguments_path = ["tool_calls", 0, "function", "arguments"]
    cases = [
        (
            ["tool_calls", 0, "function", "name"],
            prompt.index(FUNCTION_BEGIN) + len(FUNCTION_BEGIN) + 2,
            2,
            "read_file",
        ),
        (
            arguments_path,
            prompt.index(PARAMETER_BEGIN + "path>") + len(PARAMETER_BEGIN),
            arguments.index("path"),
            arguments,
        ),
        (
            arguments_path,
            prompt.index("\n/tmp/a.txt\n") + 1,
            arguments.index("/tmp/a.txt"),
            arguments,
        ),
        (
            arguments_path,
            prompt.index(PARAMETER_BEGIN + "limit>") + len(PARAMETER_BEGIN),
            arguments.index("limit"),
            arguments,
        ),
        (
            arguments_path,
            prompt.index("\n1\n") + 1,
            arguments.rindex("1"),
            arguments,
        ),
    ]
    for expected_path, templated_index, expected_char_index, channel_text in cases:
        location = build_messages_location(applied, templated_index)
        assert location["path_keys"] == [1] + expected_path
        assert location["char_index"] == expected_char_index
        assert (
            location["left5"]
            == channel_text[max(0, expected_char_index - 5) : expected_char_index]
        )
        assert (
            location["right5"]
            == channel_text[expected_char_index : expected_char_index + 5]
        )
        assert build_templated_char_index(applied, location) == templated_index


def test_structured_tool_call_location_translates_from_default_template():
    arguments = '{"path": "/tmp/a.txt", "limit": 1}'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read_file", "arguments": arguments}}],
    }
    default_applied = dict(
        DefaultResponseTemplate().apply(message), message_index=1, message=message
    )
    name_mapping = next(
        mapping
        for mapping in default_applied["key_path_prompt_mapping"]
        if mapping["key_path"] == ["tool_calls", 0, "function", "name"]
    )
    arguments_mapping = next(
        mapping
        for mapping in default_applied["key_path_prompt_mapping"]
        if mapping["key_path"] == ["tool_calls", 0, "function", "arguments"]
    )

    for template in (Qwen3p5ResponseTemplate(), Step3p5ResponseTemplate()):
        applied = dict(template.apply(message), message_index=1, message=message)
        for default_index in [
            name_mapping["text_start"] + 2,
            arguments_mapping["text_start"],
            default_applied["templated_prompt"].rindex("1"),
            arguments_mapping["text_end"],
        ]:
            default_location = build_messages_location(default_applied, default_index)
            templated_index = build_templated_char_index(applied, default_location)
            assert build_messages_location(applied, templated_index) == default_location

        empty_message = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
        }
        empty_applied = dict(
            template.apply(empty_message), message_index=1, message=empty_message
        )
        empty_location = {
            "path_keys": [1, "tool_calls", 0, "function", "arguments"],
            "char_index": 2,
            "left5": "{}",
            "right5": "",
        }
        templated_index = build_templated_char_index(empty_applied, empty_location)
        assert build_messages_location(empty_applied, templated_index) == empty_location


def test_structured_tool_call_location_aligns_escaped_argument_text():
    arguments = '{"text": "a\\nMID\\nb", "limit": 1}'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read", "arguments": arguments}}],
    }
    template = Qwen3p5ResponseTemplate()
    applied = dict(template.apply(message), message_index=1, message=message)
    templated_index = applied["templated_prompt"].index("a\nMID\nb") + 2
    location = build_messages_location(applied, templated_index)

    assert location["path_keys"] == [1, "tool_calls", 0, "function", "arguments"]
    assert location["char_index"] == arguments.index("\\") + 2
    assert location["right5"] == arguments[location["char_index"] :][:5]
    inverse_index = build_templated_char_index(applied, location)
    assert inverse_index == templated_index


def test_token_level_tool_location_preserves_raw_argument_channel():
    rejected_arguments = json.dumps(
        {"path": "/tmp/a.txt", "limit": 1}, separators=(",", ":")
    )
    chosen_arguments = json.dumps(
        {"path": "/tmp/a.txt", "limit": 10}, separators=(",", ":")
    )
    rejected_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "index": 0,
                "id": "functions.read_file:0",
                "function": {
                    "name": "read_file",
                    "arguments": rejected_arguments,
                },
            }
        ],
        "finish_reason": "tool_calls",
    }
    chosen_message = deepcopy(rejected_message)
    chosen_message["tool_calls"][0]["function"]["arguments"] = chosen_arguments
    panda_tree = PandaTree(
        {
            "version": "2.0",
            "update_time": 0,
            "dialogs": {
                "1": {
                    "messages": [{"role": "user", "content": "q"}, rejected_message],
                    "annotate": {"is_good": False},
                    "operations": [{"is_new_generated": True}],
                },
                "2": {
                    "messages": [{"role": "user", "content": "q"}, chosen_message],
                    "annotate": {"is_good": True},
                    "operations": [{"parent": "1"}],
                },
            },
        },
        tokenizer=utf8_tokenizer,
    )
    template = Qwen3p5ResponseTemplate()
    token_level = panda_tree.build_token_level_supervision_data_v1(
        tokenizer=utf8_tokenizer, response_template=template
    )[0]
    assert token_level[-1]["token_level"]["rejected_channel_text"] == rejected_arguments

    adapter = FindAndReplaceCorrectionAdapter(
        tokenizer=utf8_tokenizer,
        response_template=template,
        max_replacement_tokens=20,
    )
    correction_data = adapter.build_correction_data_from_token_level(
        token_level, is_good=False
    )
    location = correction_data[-1]["correction"]["messages_location"]
    assert location["path_keys"] == [1, "tool_calls", 0, "function", "arguments"]
    assert location["char_index"] == rejected_arguments.index("1") + 1
    assert location["right5"] == rejected_arguments[location["char_index"] :][:5]


def test_partial_json_string_boundaries_reject_closed_dangling_escape():
    assert parse_partial_json_object('{"x":"a\\u12"}') is None
    parsed = parse_partial_json_object('{"x":"a\\u1234"}')
    assert parsed["entries"][0]["value_offsets"] == [6, 7, 13]


def test_structured_location_snaps_inside_json_escape_to_left_boundary():
    arguments = '{"x":"a\\nb"}'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "f", "arguments": arguments}}],
    }
    template = Qwen3p5ResponseTemplate()
    applied = dict(template.apply(message), message_index=1, message=message)
    raw_escape_index = arguments.index("\\") + 1
    location = {
        "path_keys": [1, "tool_calls", 0, "function", "arguments"],
        "char_index": raw_escape_index,
    }
    templated_index = build_templated_char_index(applied, location)
    assert templated_index == applied["templated_prompt"].index("a\n") + 1
    assert build_messages_location(applied, templated_index)[
        "char_index"
    ] == arguments.index("\\")


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
    "template",
    [
        DefaultResponseTemplate(),
        Qwen3p5ResponseTemplate(),
        Step3p5ResponseTemplate(),
    ],
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

    empty_channel = {"role": "assistant", "content": "", "tool_calls": [{}]}
    assert template.parse(template.apply(empty_channel)["templated_prompt"])[
        "tool_calls"
    ] == [{}]

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
        "tool_calls": [{}],
    }
    assert template.parse("answer<|tool_calls|>\n") == {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [{}],
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
            "tool_calls": [{}],
        }


def test_qwen_template_preserves_open_reasoning_trailing_newline():
    template = Qwen3p5ResponseTemplate()
    message = {"role": "assistant", "reasoning": "think\n"}
    text = template.apply(message)["templated_prompt"]

    assert template.parse(text) == message


def test_step_template_uses_step_response_separators():
    template = Step3p5ResponseTemplate()
    message = {
        "role": "assistant",
        "reasoning": "think",
        "content": "answer",
        "tool_calls": [
            {
                "index": 0,
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            },
            {
                "index": 1,
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            },
        ],
        "finish_reason": "tool_calls",
    }
    text = template.apply(message)["templated_prompt"]

    assert THINK_END + "\nanswer" + TOOL_CALL_BEGIN in text
    assert TOOL_CALL_END + TOOL_CALL_BEGIN in text
    assert (
        template.apply(template.parse(text, finish_reason="tool_calls"))[
            "templated_prompt"
        ]
        == text
    )


def test_far_template_fork_keeps_the_first_duplicate_tool_call():
    trajectory = get_test_trajectories("redundant_call")
    fork_message_index = trajectory["fork_message_index"]
    rejected_message = deepcopy(trajectory["rejected_messages"][fork_message_index])
    template = Step3p5ResponseTemplate()
    applied = dict(
        template.apply(rejected_message),
        message_index=0,
        message=rejected_message,
    )
    first_call = applied["templated_prompt"].find(TOOL_CALL_BEGIN)
    fork_index = applied["templated_prompt"].find(TOOL_CALL_BEGIN, first_call + 1)
    rejected_message["token_level"] = {
        "chosen_text": "",
        "rejected_text_unicode_range": [
            fork_index,
            len(applied["templated_prompt"]),
        ],
        "messages_location": build_messages_location(applied, fork_index),
    }

    adapter = FindAndReplaceCorrectionAdapter(
        tokenizer=utf8_tokenizer,
        response_template=template,
        max_location_tokens=20,
    )
    far_text = adapter.build_correction_from_rejected_messages(
        [rejected_message], templated_char_index=fork_index
    )["find_and_replace"]["far_text"]
    split = adapter.special_tokens["split"]
    assert far_text.split(split)[1].startswith(TOOL_CALL_BEGIN)
    assert far_text.split(split)[2] == "1"

    find_and_replace = adapter.verifier.parse(far_text)["find_and_replace"]
    located = adapter.verifier.locate_templated([rejected_message], find_and_replace)
    assert located["templated_char_index"] == fork_index


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
        response_template={"name_or_path": "Qwen/Qwen3.5-35B-A3B"},
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
    [None, {"name_or_path": "Qwen/Qwen3.5-35B-A3B"}],
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
        response_template={"name_or_path": "Qwen/Qwen3.5-35B-A3B"},
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
    prefix = adapter.build_partial_templated_prompt(rejected_message, partial_message)[
        "templated_prompt"
    ]
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
                            prefix + "<tool_call>\n"
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
