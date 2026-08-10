from mximport import inpkg


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


def get_test_reasoning_tool_calls_msgs1(error_type="reasoning"):
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
    if "reasoning" in error_type or "content" in error_type or "bad_argument_value" in error_type:
        arguments = '{"path": "/tmp/b.txt", "limit": 10}'
    if "bad_argument_key" in error_type:
        arguments = '{"file": "/tmp/a.txt", "limit": 10}'
    if "bad_argument_num" in error_type:
        arguments = '{"path": "/tmp/a.txt"}'
    if "bad_argument_arg2" in error_type:
        arguments = '{"path": "/tmp/a.txt", "limit": 1}'
    if "bad_argument_json" in error_type:
        arguments = '{"path": /tmp/a.txt, "limit": 10}'
    rejected_msgs = [
        {"role": "user", "content": "Read the first 10 lines of /tmp/a.txt for me."},
        {
            "role": "assistant",
            "reasoning": f"The user wants /tmp/a.txt, so I should read /tmp/{'b' if 'reasoning' in error_type else 'a'}.txt with limit 10.",
            "content": "I will call read_file tool to read `/tmp/b.txt` with limit 10." if 'content' in error_type else "",
            "tool_calls": [
                {
                    "index": 0,
                    "type": "function",
                    "id": "functions.read_file:0",
                    "function": {
                        "name": "read" if "call_name" in error_type else "read_file",
                        "arguments": arguments,
                    },
                }
            ],
            "finish_reason": "tool_calls",
        },
    ]
    if "no_call" in error_type:
        del rejected_msgs[-1]["tool_calls"]
        rejected_msgs[-1]["content"] = "I will call read_file tool to read `/tmp/a.txt` with limit 10."
        rejected_msgs[-1]["finish_reason"] = "stop"
    if "redundant_call" in error_type:
        rejected_msgs[-1]["tool_calls"].append({
                    "index": 1,
                    "type": "function",
                    "id": "functions.read_file:1",
                    "function": {
                        "name": "read_file",
                        "arguments": arguments,
                    },
                })
    return rejected_msgs, tools


def get_test_reasoning_tool_calls_partial_msgs1():
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
        "reasoning": build_far_ref("b.txt", "a.txt"),
        "content": build_far_ref("`/tmp/b.txt`", "`/tmp/a.txt`"),
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
        "redundant_call": build_far_ref('\n' + CALL_BEGIN_MARKER, stop, -1),
    }
    expected_location_paths = {
        "reasoning": [1, "reasoning"],
        "content": [1, "content"],
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
        partials["error_type:" + error_type] = {
            "default_far_ref": default_far_ref,
            "partial_message": apply_result["partial_messages"][-1],
        }

    reasoning = partials["error_type:reasoning"]["partial_message"]
    assert reasoning == {
        "role": "assistant",
        "reasoning": "The user wants /tmp/a.txt, so I should read /tmp/a.txt",
    }, reasoning

    content = partials["error_type:content"]["partial_message"]
    assert content["content"] == "I will call read_file tool to read `/tmp/a.txt`"
    assert "tool_calls" not in content and "finish_reason" not in content, content

    call_name = partials["error_type:call_name"]["partial_message"]
    assert call_name["tool_calls"][0]["function"] == {"name": "read_file"}
    assert "finish_reason" not in call_name, call_name

    bad_argument_value = partials["error_type:bad_argument_value"][
        "partial_message"
    ]
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
