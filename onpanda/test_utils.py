import json
from copy import deepcopy

from mximport import inpkg

_REASONING_ERROR_TYPES = (
    "stop_reasoning",
    "resume_reasoning",
    "stop_content",
    "resume_content",
)

ERROR_TYPES = _REASONING_ERROR_TYPES + (
    "bad_reasoning",
    "bad_content",
    "call_name",
    "bad_argument_value",
    "bad_argument_key",
    "bad_argument_num",
    "bad_argument_arg2",
    "bad_argument_json",
    "no_call",
    "redundant_call",
)


def build_test_tokenizer(name_or_path="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"):
    with inpkg():
        from .token_level_supervision_utils import _from_pretrained_local_first

    return _from_pretrained_local_first(name_or_path, use_fast=True)


def get_test_rejected_msgs1():
    rejected_msgs1 = [
        {"role": "user", "content": "Name three kinds of fruit:"},
        {
            "role": "assistant",
            "content": "Apple, potato, banana.",
            "finish_reason": "stop",
            "token_level": {
                "chosen_text": " orange",
                "rejected_text": " potato",
                "chosen_text_unicode_range": [6, 13],  # " potato" start at index 6
                "rejected_text_unicode_range": [6, 13],
                "version": "1.0",
                "chosen_dialog_key": 2,
                "rejected_dialog_key": 1,
                "rejected_finish_reason": "stop",
            },
        },
    ]
    far_text_gt1 = "<|fim_pad|> potato<|fim_pad|>0<|fim_pad|> orange<|fim_pad|>"
    return rejected_msgs1, far_text_gt1


def get_test_reasoning_msgs1(error_type="stop_reasoning"):
    """A rejected reasoning response exercising channel-ending and resumption corrections."""
    rejected_message = {"role": "assistant"}
    if error_type == "stop_reasoning":
        rejected_message.update(
            reasoning="2 × 5 + 4 = 10 + 4 = 14.wait!wait!wait!wait!",
            content="The answer is 14.",
            finish_reason="stop",
        )
    elif error_type == "resume_reasoning":
        rejected_message.update(
            reasoning="2 × 5 + 4 = 10 + 4 =",
            content="The answer is 14.",
            finish_reason="stop",
        )
    elif error_type == "stop_content":
        rejected_message.update(
            reasoning="2 × 5 + 4 = 10 + 4 = 14.",
            content="The answer is 14.wait!wait!wait!wait!",
            finish_reason="stop",
        )
    elif error_type == "resume_content":
        rejected_message.update(
            reasoning="2 × 5 + 4 = 10 + 4 = 14.",
            content="The answer is",
            finish_reason="stop",
        )
    return [
        {"role": "user", "content": "Calculate 2 × 5 + 4."},
        rejected_message,
    ]


def get_test_reasoning_partial_msgs_all():
    """Build reference FAR partials for reasoning and content boundaries."""
    with inpkg():
        from .correcting_model.far_correction_utils import (
            FindAndReplaceCorrectionAdapter,
        )
        from .response_templates import DefaultResponseTemplate

    adapter = FindAndReplaceCorrectionAdapter(
        response_template=DefaultResponseTemplate(),
        max_replacement_tokens=20,
    )
    split = adapter.special_tokens["split"]
    stop = adapter.special_tokens["stop"]

    def build_far_ref(location_text, replacement_token, location_index=0):
        return (
            f"{split}{location_text}{split}{location_index}"
            f"{split}{replacement_token}{split}"
        )

    default_far_refs = {
        "stop_reasoning": build_far_ref("wait!", adapter.special_tokens["reasoning"]),
        "resume_reasoning": build_far_ref(
            adapter.response_template.reasoning_end_marker, " 14"
        ),
        "stop_content": build_far_ref("wait!", stop),
        "resume_content": build_far_ref(stop, " 14"),
    }
    expected_location_paths = {
        "stop_reasoning": [1, "reasoning"],
        "resume_reasoning": [1, "reasoning"],
        "stop_content": [1, "content"],
        "resume_content": [1, "content"],
    }

    partials = {}
    for error_type, default_far_ref in default_far_refs.items():
        messages = get_test_reasoning_msgs1(error_type)
        apply_result = adapter.apply(messages, default_far_ref)
        location = apply_result["correction"]["messages_location"]
        assert location.get("path_keys") == expected_location_paths[error_type], (
            error_type,
            location,
        )
        assert not location.get("not_found"), (error_type, location)
        error_key = "error_type:" + error_type
        partials[error_key] = {
            "rejected_messages": messages,
            "default_far_ref": default_far_ref,
            "partial_message": apply_result["partial_messages"][-1],
        }

    assert partials["error_type:stop_reasoning"]["partial_message"] == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
        "content": "",
        "finish_reason": "reasoning_end",
    }
    assert partials["error_type:resume_reasoning"]["partial_message"] == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14",
    }
    assert partials["error_type:stop_content"]["partial_message"] == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
        "content": "The answer is 14.",
        "finish_reason": "stop",
    }
    assert partials["error_type:resume_content"]["partial_message"] == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
        "content": "The answer is 14",
    }
    return partials


def get_test_reasoning_tool_calls_msgs1(error_type="bad_reasoning"):
    """A rejected reasoning tool call response, whose thinking picks the wrong file path."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file on the user's computer and return its content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                        "limit": {"type": "integer", "description": "Max line number"},
                    },
                    "required": ["path"],
                },
            },
        }
    ]
    arguments = '{"path": "/tmp/a.txt", "limit": 10}'
    if error_type in ("bad_reasoning", "bad_content", "bad_argument_value"):
        arguments = '{"path": "/tmp/b.txt", "limit": 10}'
    elif error_type == "bad_argument_key":
        arguments = '{"file": "/tmp/a.txt", "limit": 10}'
    elif error_type == "bad_argument_num":
        arguments = '{"path": "/tmp/a.txt"}'
    elif error_type == "bad_argument_arg2":
        arguments = '{"path": "/tmp/a.txt", "limit": 1}'
    elif error_type == "bad_argument_json":
        arguments = '{"path": /tmp/a.txt, "limit": 10}'
    rejected_msgs = [
        {"role": "user", "content": "Read the first 10 lines of /tmp/a.txt for me."},
        {
            "role": "assistant",
            "reasoning": (
                f"The user wants /tmp/a.txt, so I should read /tmp/"
                f"{'b' if error_type == 'bad_reasoning' else 'a'}.txt with limit 10."
            ),
            "content": (
                "I will call read_file tool to read `/tmp/b.txt` with limit 10."
                if error_type == "bad_content"
                else ""
            ),
            "tool_calls": [
                {
                    "index": 0,
                    "type": "function",
                    "id": "functions.read_file:0",
                    "function": {
                        "name": "read" if error_type == "call_name" else "read_file",
                        "arguments": arguments,
                    },
                }
            ],
            "finish_reason": "tool_calls",
        },
    ]
    if error_type == "no_call":
        del rejected_msgs[-1]["tool_calls"]
        rejected_msgs[-1][
            "content"
        ] = "I will call read_file tool to read `/tmp/a.txt` with limit 10."
        rejected_msgs[-1]["finish_reason"] = "stop"
    if error_type == "redundant_call":
        rejected_msgs[-1]["tool_calls"].append(
            {
                "index": 1,
                "type": "function",
                "id": "functions.read_file:1",
                "function": {
                    "name": "read_file",
                    "arguments": arguments,
                },
            }
        )
    return rejected_msgs, tools


def get_test_reasoning_tool_calls_partial_msgs_all():
    """Build reference FAR partials in the canonical default response template."""
    with inpkg():
        from .correcting_model.far_correction_utils import (
            FindAndReplaceCorrectionAdapter,
        )
        from .response_templates import DefaultResponseTemplate
        from .response_templates.default import (
            CALL_BEGIN_MARKER,
            CALL_FIELD_MARKERS,
            TOOL_CALLS_MARKER,
        )

    adapter = FindAndReplaceCorrectionAdapter(
        response_template=DefaultResponseTemplate(),
        max_replacement_tokens=20,
    )
    split = adapter.special_tokens["split"]
    stop = adapter.special_tokens["stop"]
    call_arguments_marker = dict(
        (tuple(key_path), marker) for key_path, marker in CALL_FIELD_MARKERS
    )[("function", "arguments")]

    def build_far_ref(location_text, replacement_token, location_index=0):
        return (
            f"{split}{location_text}{split}{location_index}"
            f"{split}{replacement_token}{split}"
        )

    default_far_refs = {
        "bad_reasoning": build_far_ref("b.txt", "a.txt"),
        "bad_content": build_far_ref("`/tmp/b.txt`", "`/tmp/a.txt`"),
        "call_name": build_far_ref(
            "read\n" + call_arguments_marker,
            "read_file",
        ),
        "bad_argument_value": build_far_ref("b.txt", "a.txt"),
        "bad_argument_key": build_far_ref('"file"', '"path"'),
        "bad_argument_num": build_far_ref("}" + stop, ', "limit"'),
        "bad_argument_arg2": build_far_ref("1}" + stop, "10"),
        "bad_argument_json": build_far_ref(
            '/tmp/a.txt, "limit": 10}',
            '"/tmp/a.txt"',
        ),
        "no_call": build_far_ref(stop, TOOL_CALLS_MARKER, -1),
        "redundant_call": build_far_ref("\n" + CALL_BEGIN_MARKER, stop, -1),
    }
    expected_location_paths = {
        "bad_reasoning": [1, "reasoning"],
        "bad_content": [1, "content"],
        "call_name": [1, "tool_calls", 0, "function", "name"],
        "bad_argument_value": [
            1,
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_key": [
            1,
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_num": [
            1,
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_arg2": [
            1,
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_json": [
            1,
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        # Template scaffolding maps to the channel immediately before the marker.
        "no_call": [1, "content"],
        "redundant_call": [
            1,
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
    }

    partials = {}
    for error_type, default_far_ref in default_far_refs.items():
        messages, tools = get_test_reasoning_tool_calls_msgs1(error_type)
        apply_result = adapter.apply(messages, default_far_ref, tools=tools)
        location = apply_result["correction"]["messages_location"]
        assert location.get("path_keys") == expected_location_paths[error_type], (
            error_type,
            location,
        )
        assert not location.get("not_found"), (error_type, location)
        error_key = "error_type:" + error_type
        partials[error_key] = {
            "rejected_messages": messages,
            "default_far_ref": default_far_ref,
            "partial_message": apply_result["partial_messages"][-1],
        }

    reasoning = partials["error_type:bad_reasoning"]["partial_message"]
    assert reasoning == {
        "role": "assistant",
        "reasoning": "The user wants /tmp/a.txt, so I should read /tmp/a.txt",
    }, reasoning

    content = partials["error_type:bad_content"]["partial_message"]
    assert content["content"] == "I will call read_file tool to read `/tmp/a.txt`"
    assert "tool_calls" not in content and "finish_reason" not in content, content

    call_name = partials["error_type:call_name"]["partial_message"]
    assert call_name["tool_calls"][0]["function"] == {"name": "read_file"}
    assert "finish_reason" not in call_name, call_name

    bad_argument_value = partials["error_type:bad_argument_value"]["partial_message"]
    assert bad_argument_value["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt'
    ), bad_argument_value

    bad_argument_key = partials["error_type:bad_argument_key"]["partial_message"]
    assert bad_argument_key["tool_calls"][0]["function"]["arguments"] == (
        '{"path"'
    ), bad_argument_key

    bad_argument_num = partials["error_type:bad_argument_num"]["partial_message"]
    assert bad_argument_num["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt", "limit"'
    ), bad_argument_num

    bad_argument_arg2 = partials["error_type:bad_argument_arg2"]["partial_message"]
    assert bad_argument_arg2["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt", "limit": 10'
    ), bad_argument_arg2

    bad_argument_json = partials["error_type:bad_argument_json"]["partial_message"]
    assert bad_argument_json["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt"'
    ), bad_argument_json

    no_call = partials["error_type:no_call"]["partial_message"]
    assert no_call["tool_calls"] == [], no_call
    assert no_call["content"].endswith("`/tmp/a.txt` with limit 10."), no_call
    assert "finish_reason" not in no_call, no_call

    redundant_call = partials["error_type:redundant_call"]["partial_message"]
    assert len(redundant_call["tool_calls"]) == 1, redundant_call
    assert redundant_call["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt", "limit": 10}'
    ), redundant_call
    assert redundant_call["finish_reason"] == "tool_calls", redundant_call

    return partials


def get_test_msgs(error_type):
    if error_type in _REASONING_ERROR_TYPES:
        return get_test_reasoning_msgs1(error_type), None
    return get_test_reasoning_tool_calls_msgs1(error_type)


def get_test_partial_msgs_all():
    return {
        **get_test_reasoning_partial_msgs_all(),
        **get_test_reasoning_tool_calls_partial_msgs_all(),
    }


def print_response_template_partial_messages(response_template):
    with inpkg():
        from .correcting_model.far_correction_utils import (
            FindAndReplaceCorrectionAdapter,
        )
        from .response_templates import build_messages_location
        from .token_level_supervision_utils import compute_token_level_supervision

    adapter = FindAndReplaceCorrectionAdapter(
        response_template=response_template,
        tokenizer=response_template.name_or_path or "utf8_tokenizer",
        max_replacement_tokens=1,
    )
    for error_key, partial_ref in get_test_partial_msgs_all().items():
        rejected_message = partial_ref["rejected_messages"][-1]
        partial_message = partial_ref["partial_message"]
        partial_templated = response_template.apply(partial_message)["templated_prompt"]
        rejected_applied = response_template.apply(rejected_message)
        rejected_templated = rejected_applied["templated_prompt"]
        partial_result = adapter.build_partial_templated_prompt(
            rejected_message, partial_message
        )
        replacement_tokens_1 = partial_result["templated_prompt"]
        token_level = compute_token_level_supervision(
            chosen_content=partial_templated,
            rejected_content=rejected_templated,
            tokenizer=adapter.tokenizer,
        )
        token_level["chosen_text"] = partial_result["replacement"]
        token_level["messages_location"] = build_messages_location(
            dict(
                message_index=0,
                message=rejected_message,
                **rejected_applied,
            ),
            token_level["rejected_text_unicode_range"][0],
        )
        if token_level["rejected_text"]:
            templated_message = deepcopy(rejected_message)
            templated_message["token_level"] = token_level
            templated_far = adapter.build_correction_from_rejected_messages(
                [templated_message],
                templated_char_index=token_level["rejected_text_unicode_range"][0],
            )["find_and_replace"]["far_text"]
        else:
            split = adapter.special_tokens["split"]
            stop = adapter.special_tokens["stop"]
            templated_far = (
                f"{split}{stop}{split}0{split}"
                f"{partial_result['replacement'] or stop}{split}"
            )
        partial_templated, replacement_tokens_1 = (
            text if len(text) <= 40 else "..." + text[-37:]
            for text in (partial_templated, replacement_tokens_1)
        )
        print(
            "\x1b[31m%s\x1b[0m"
            % f"\nERROR_TYPE: {error_key.removeprefix('error_type:')}"
        )
        print("rejected_response:")
        print(json.dumps(rejected_message, ensure_ascii=False, indent=2)[2:-2])
        print(
            f"default_far_ref: {partial_ref['default_far_ref']!r}",
            f"partial____templated: '''{partial_templated}'''",
            f"replacement_tokens_1: '''{replacement_tokens_1}'''",
            f"templated_far: {templated_far!r}",
            sep="\n",
        )
        print()


if __name__ == "__main__":
    from boxx import *

    partial_msgs_all = get_test_partial_msgs_all()
