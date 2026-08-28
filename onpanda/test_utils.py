import json
from copy import deepcopy

from mximport import inpkg

_REASONING_ERROR_TYPES = (
    "stop_reasoning",
    "resume_reasoning",
    "stop_content",
    "resume_content",
    "bad_previous_turn",
    "is_good",
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


def build_test_tokenizer(name_or_path="Qwen/Qwen3.5-35B-A3B"):
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


def get_test_far_text_cases(adapter=None):
    """Build shared legacy FAR and trajectory verification cases.

    Each case's ``gt_far`` field stores the ground-truth FAR text.
    """
    if adapter is None:
        with inpkg():
            from .correcting_model.far_correction_utils import (
                FindAndReplaceCorrectionAdapter,
            )

        adapter = FindAndReplaceCorrectionAdapter(
            tokenizer=build_test_tokenizer(),
            special_tokens=dict(
                split="<|fim_pad|>",
                stop="<|fim_suffix|>",
                is_good="<|fim_prefix|>",
                reasoning="<|fim_middle|>",
            ),
        )

    rejected_messages, gt_far = get_test_rejected_msgs1()
    gt_apply = adapter.apply(rejected_messages, gt_far)
    gt_correction = deepcopy(gt_apply["correction"])
    gt_correction.update(
        status="partial",
        fork_message_index=gt_correction["messages_location"]["path_keys"][0],
        corrected_messages=gt_apply["partial_messages"],
    )
    gt_trajectory = dict(
        messages=deepcopy(rejected_messages),
        tools=None,
        corrections=[gt_correction],
    )

    split = adapter.special_tokens["split"]
    is_good = adapter.special_tokens["is_good"]
    location_text = gt_correction["find_and_replace"]["location_text"]
    location_index = gt_correction["find_and_replace"]["location_index"]
    replacement_token = gt_correction["find_and_replace"]["replacement_token"]
    far_specs = [
        (
            "case1_all_correct",
            gt_far,
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=1.0,
                is_good_cls_reward=1.0,
            ),
        ),
        (
            "case2_wrong_replacement",
            f"{split}{location_text}{split}{location_index}{split} banana{split}",
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=0.0,
                is_good_cls_reward=1.0,
            ),
        ),
        (
            "case3_wrong_location_index",
            f"{split} {split}{location_index + 1}{split}{replacement_token}{split}",
            dict(
                format_reward=1.0,
                location_reward=0.0,
                replacement_reward=0.0,
                is_good_cls_reward=1.0,
            ),
        ),
        (
            "case4_is_good_prediction",
            f"{split}{is_good}{split}",
            dict(
                format_reward=1.0,
                location_reward=0.0,
                replacement_reward=0.0,
                is_good_cls_reward=0.0,
            ),
        ),
        (
            "case5_bad_format_missing_end_split",
            f"{split}{location_text}{split}{location_index}{split}{replacement_token}",
            dict(
                format_reward=0.0,
                location_reward=0.0,
                replacement_reward=0.0,
                is_good_cls_reward=0.0,
            ),
        ),
        (
            "case6_parse_success_but_locate_not_found",
            f"{split} no_such_text{split}0{split}{replacement_token}{split}",
            dict(
                format_reward=0.5,
                location_reward=0.0,
                replacement_reward=0.0,
                is_good_cls_reward=1.0,
            ),
        ),
        (
            "case7_loose_token_reward",
            f"{split}potato, banana{split}{location_index}{split}orange, pineapple{split}",
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=1.0,
                is_good_cls_reward=1.0,
            ),
        ),
        (
            "case8_same_prefix",
            f"{split}Apple, potato{split}{location_index}{split}Apple, orange, pineapple.<|fim_suffix|>{split}",
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=1.0,
                is_good_cls_reward=1.0,
            ),
        ),
    ]

    cases = []
    for name, pred_far, expected_rewards in far_specs:
        pred_apply = adapter.apply(rejected_messages, pred_far)
        pred_correction = deepcopy(pred_apply["correction"])
        if pred_correction["find_and_replace"].get("is_good"):
            pred_correction.update(status="is_good", fork_message_index=None)
        elif pred_correction["reward_with_feedback"]["parse_reward"] == 0.0:
            pred_correction["status"] = "parse_failed"
        elif pred_correction["messages_location"].get("not_found"):
            pred_correction["status"] = "not_found"
        else:
            pred_correction.update(
                status="partial",
                fork_message_index=pred_correction["messages_location"]["path_keys"][0],
                corrected_messages=pred_apply["partial_messages"],
            )
        cases.append(
            dict(
                name=name,
                pred_far=pred_far,
                gt_far=gt_far,
                expected_rewards=expected_rewards,
                pred_trajectory=dict(
                    messages=deepcopy(rejected_messages),
                    tools=None,
                    corrections=[pred_correction],
                ),
                gt_trajectory=deepcopy(gt_trajectory),
            )
        )
    return cases


def get_test_reasoning_msgs1(error_type="stop_reasoning"):
    """Build rejected reasoning responses for channel and historical-turn corrections."""
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
    elif error_type == "bad_previous_turn":
        rejected_message.update(
            reasoning="2 × 5 + 4 = 2 × 9 = 18.",
            content="The answer is 18.",
            finish_reason="stop",
        )

    messages = [
        {"role": "user", "content": "Calculate 2 × 5 + 4."},
        rejected_message,
    ]
    if error_type == "bad_previous_turn":
        messages.extend(
            [
                {"role": "user", "content": "Wrong! Do multiplication first"},
                {
                    "role": "assistant",
                    "reasoning": (
                        "The user pointed out my mistake in the last round, so I will "
                        "recalculate.. 2 × 5 + 4 = 10 + 4  = 14."
                    ),
                    "content": "You are right, the corrected answer is 14.",
                    "finish_reason": "stop",
                },
            ]
        )
    return messages


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


def get_test_trajectories(error_type=None):
    """Build test trajectories and their reference correction data."""
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
        "stop_reasoning": build_far_ref("wait!", adapter.special_tokens["reasoning"]),
        "resume_reasoning": build_far_ref(
            adapter.response_template.reasoning_end_marker, " 14"
        ),
        "stop_content": build_far_ref("wait!", stop),
        "resume_content": build_far_ref(stop, " 14"),
        "bad_previous_turn": build_far_ref("2 × 9 = 18.", "10 + 4 = 14."),
        "is_good": None,
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
        "stop_reasoning": ["reasoning"],
        "resume_reasoning": ["reasoning"],
        "stop_content": ["content"],
        "resume_content": ["content"],
        "bad_reasoning": ["reasoning"],
        "bad_content": ["content"],
        "bad_previous_turn": ["reasoning"],
        "call_name": ["tool_calls", 0, "function", "name"],
        "bad_argument_value": [
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_key": [
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_num": [
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_arg2": [
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        "bad_argument_json": [
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
        # Template scaffolding maps to the channel immediately before the marker.
        "no_call": ["content"],
        "redundant_call": [
            "tool_calls",
            0,
            "function",
            "arguments",
        ],
    }

    corrected_reasoning_messages = [
        {"role": "user", "content": "Calculate 2 × 5 + 4."},
        {
            "role": "assistant",
            "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
            "content": "The answer is 14.",
            "finish_reason": "stop",
        },
    ]
    corrected_tool_messages = [
        {"role": "user", "content": "Read the first 10 lines of /tmp/a.txt for me."},
        {
            "role": "assistant",
            "reasoning": (
                "The user wants /tmp/a.txt, so I should read /tmp/a.txt with limit 10."
            ),
            "content": "",
            "tool_calls": [
                {
                    "index": 0,
                    "type": "function",
                    "id": "functions.read_file:0",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/a.txt", "limit": 10}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
        },
    ]

    trajectories = {}
    for current_error_type in ERROR_TYPES:
        default_far_ref = default_far_refs[current_error_type]
        if current_error_type in _REASONING_ERROR_TYPES:
            if current_error_type == "is_good":
                # An explicit None fork marks a no-op correction.
                messages = deepcopy(corrected_reasoning_messages)
                trajectories["error_type:" + current_error_type] = {
                    "messages": messages,
                    "corrections": [
                        {
                            "fork_message_index": None,
                        }
                    ],
                }
                continue
            messages = get_test_reasoning_msgs1(current_error_type)
            corrected_messages = deepcopy(corrected_reasoning_messages)
            tools = None
        else:
            messages, tools = get_test_reasoning_tool_calls_msgs1(current_error_type)
            corrected_messages = deepcopy(corrected_tool_messages)
            # Keep the corrected content because these forks precede the tool-call marker.
            if current_error_type in ("bad_content", "no_call"):
                corrected_messages[1][
                    "content"
                ] = "I will call read_file tool to read `/tmp/a.txt` with limit 10."

        trajectory = {
            "messages": messages,
            "corrections": [
                {
                    "fork_message_index": 1,
                    "corrected_messages": corrected_messages,
                    "default_far_ref": default_far_ref,
                }
            ],
        }
        correction = trajectory["corrections"][0]
        if tools is not None:
            trajectory["tools"] = tools

        apply_result = adapter.apply(
            messages,
            correction["default_far_ref"],
            tools=trajectory.get("tools"),
        )
        location = apply_result["correction"]["messages_location"]
        assert (
            location.get("path_keys")
            == [correction["fork_message_index"]]
            + expected_location_paths[current_error_type]
        ), (
            current_error_type,
            location,
        )
        assert not location.get("not_found"), (current_error_type, location)
        correction["partial_messages"] = apply_result["partial_messages"]
        fork_message_index = correction["fork_message_index"]
        partial_templated = adapter.response_template.apply(
            correction["partial_messages"][fork_message_index]
        )["templated_prompt"]
        corrected_templated = adapter.response_template.apply(
            correction["corrected_messages"][fork_message_index]
        )["templated_prompt"]
        assert corrected_templated.startswith(partial_templated), current_error_type
        trajectories["error_type:" + current_error_type] = trajectory

    def partial_message(current_error_type):
        trajectory = trajectories["error_type:" + current_error_type]
        correction = trajectory["corrections"][0]
        return correction["partial_messages"][correction["fork_message_index"]]

    assert partial_message("stop_reasoning") == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
        "content": "",
        "finish_reason": "reasoning_end",
    }
    assert partial_message("resume_reasoning") == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14",
    }
    assert partial_message("stop_content") == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
        "content": "The answer is 14.",
        "finish_reason": "stop",
    }
    assert partial_message("resume_content") == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
        "content": "The answer is 14",
    }

    bad_previous_turn = partial_message("bad_previous_turn")
    assert bad_previous_turn == {
        "role": "assistant",
        "reasoning": "2 × 5 + 4 = 10 + 4 = 14.",
    }, bad_previous_turn

    reasoning = partial_message("bad_reasoning")
    assert reasoning == {
        "role": "assistant",
        "reasoning": "The user wants /tmp/a.txt, so I should read /tmp/a.txt",
    }, reasoning

    content = partial_message("bad_content")
    assert content["content"] == "I will call read_file tool to read `/tmp/a.txt`"
    assert "tool_calls" not in content and "finish_reason" not in content, content

    call_name = partial_message("call_name")
    assert call_name["tool_calls"][0]["function"] == {"name": "read_file"}
    assert "finish_reason" not in call_name, call_name

    bad_argument_value = partial_message("bad_argument_value")
    assert bad_argument_value["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt'
    ), bad_argument_value

    bad_argument_key = partial_message("bad_argument_key")
    assert bad_argument_key["tool_calls"][0]["function"]["arguments"] == (
        '{"path"'
    ), bad_argument_key

    bad_argument_num = partial_message("bad_argument_num")
    assert bad_argument_num["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt", "limit"'
    ), bad_argument_num

    bad_argument_arg2 = partial_message("bad_argument_arg2")
    assert bad_argument_arg2["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt", "limit": 10'
    ), bad_argument_arg2

    bad_argument_json = partial_message("bad_argument_json")
    assert bad_argument_json["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt"'
    ), bad_argument_json

    no_call = partial_message("no_call")
    assert no_call["tool_calls"] == [{}], no_call
    assert no_call["content"].endswith("`/tmp/a.txt` with limit 10."), no_call
    assert "finish_reason" not in no_call, no_call

    redundant_call = partial_message("redundant_call")
    assert len(redundant_call["tool_calls"]) == 1, redundant_call
    assert redundant_call["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "/tmp/a.txt", "limit": 10}'
    ), redundant_call
    assert redundant_call["finish_reason"] == "tool_calls", redundant_call

    if error_type is not None:
        return trajectories["error_type:" + error_type]
    return trajectories


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
    for error_key, trajectory in get_test_trajectories().items():
        for correction_index, correction in enumerate(trajectory["corrections"]):
            message_index = correction["fork_message_index"]
            if message_index is None:
                continue
            rejected_message = trajectory["messages"][message_index]
            partial_message = correction["partial_messages"][message_index]
            partial_templated = response_template.apply(partial_message)[
                "templated_prompt"
            ]
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
            correction_label = (
                f"{error_key.removeprefix('error_type:')}"
                if len(trajectory["corrections"]) == 1
                else f"{error_key.removeprefix('error_type:')}[{correction_index}]"
            )
            print("\x1b[31m%s\x1b[0m" % f"\nERROR_TYPE: {correction_label}")
            print("rejected_response:")
            print(json.dumps(rejected_message, ensure_ascii=False, indent=2)[2:-2])
            print(
                f"default_far_ref: {correction['default_far_ref']!r}",
                f"partial____templated: '''{partial_templated}'''",
                f"replacement_tokens_1: '''{replacement_tokens_1}'''",
                f"templated_far: {templated_far!r}",
                sep="\n",
            )
            print()


if __name__ == "__main__":
    from boxx import *

    trajectories = get_test_trajectories()
    tree(trajectories)
    trajectory = trajectories["error_type:is_good"]
    assert trajectory["corrections"][0]["fork_message_index"] is None
