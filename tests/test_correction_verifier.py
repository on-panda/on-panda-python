from copy import deepcopy

import pytest

from onpanda import CorrectionVerifier, utf8_tokenizer
from onpanda.correcting_model.far_text_parse import FindAndReplaceCodecMixin


class BoundaryTokenizer:
    """Small tokenizer that merges a token only when continuation text is present."""

    name_or_path = "boundary-tokenizer"

    def __init__(self):
        self.tools_seen = []

    def encode(self, text, **kwargs):
        tokens = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("abcdef", cursor):
                tokens.append(1001)
                cursor += len("abcdef")
            elif text.startswith("abcde", cursor):
                tokens.append(1002)
                cursor += len("abcde")
            elif text.startswith("=read", cursor):
                tokens.append(1000)
                cursor += len("=read")
            else:
                tokens.append(ord(text[cursor]))
                cursor += 1
        return tokens

    def decode(self, tokens, **kwargs):
        return "".join(
            {
                1000: "=read",
                1001: "abcdef",
                1002: "abcde",
            }.get(token, chr(token))
            for token in tokens
        )

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.tools_seen.append(kwargs.get("tools", "missing"))
        rendered = "".join(message.get("content") or "" for message in messages)
        return self.encode(rendered) if tokenize else rendered


def build_trajectory(base_messages, corrected_messages, fork_message_index=1, **extra):
    correction = dict(
        fork_message_index=fork_message_index,
        corrected_messages=corrected_messages,
        **extra,
    )
    return dict(messages=deepcopy(base_messages), tools=None, corrections=[correction])


def test_verify_one_accepts_partial_prediction_covering_truncated_gt_tokens():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "good", "finish_reason": "stop"},
        ],
    )
    pred = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "go"}],
    )

    result = CorrectionVerifier(
        tokenizer=utf8_tokenizer, max_replacement_tokens=2
    ).verify_one(pred, gt)
    rewards = result["reward_with_feedback"]
    assert rewards["final_reward"] == 1.0
    assert rewards["is_good_cls_reward"] == 1.0
    assert rewards["fork_message_index_reward"] == 1.0
    assert rewards["tokenizer"] == {"name_or_path": "onpanda.UTF8Tokenizer"}
    assert result["find_and_replace"] == {}
    assert result["messages_location"] == {}


def test_verify_max_selects_the_best_equal_candidate_pair():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [
            {
                "fork_message_index": 1,
                "corrected_messages": [
                    base[0],
                    {"role": "assistant", "content": "good", "finish_reason": "stop"},
                ],
            },
            {
                "fork_message_index": 1,
                "corrected_messages": [
                    base[0],
                    {"role": "assistant", "content": "better", "finish_reason": "stop"},
                ],
            },
        ],
    }
    pred = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [
            {
                "fork_message_index": 1,
                "corrected_messages": [
                    base[0],
                    {"role": "assistant", "content": "wrong", "finish_reason": "stop"},
                ],
            },
            {
                "fork_message_index": 1,
                "corrected_messages": [
                    base[0],
                    {"role": "assistant", "content": "better"},
                ],
            },
        ],
    }

    verifier = CorrectionVerifier(
        tokenizer=utf8_tokenizer,
        max_replacement_tokens=2,
        multi_correction_gt_mode="max",
    )
    results = verifier.verify(pred, gt)
    assert len(results) == 1
    rewards = results[0]["reward_with_feedback"]
    assert rewards["final_reward"] == 1.0
    assert rewards["pred_correction_index"] == 1
    assert rewards["gt_correction_index"] == 1
    assert results[0]["find_and_replace"] == {}
    assert results[0]["messages_location"] == {}


def test_verify_requires_tools_to_match_the_base_identity():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    corrected = [
        base[0],
        {"role": "assistant", "content": "good", "finish_reason": "stop"},
    ]
    pred = build_trajectory(base, corrected)
    gt = build_trajectory(base, corrected)
    gt["tools"] = [{"type": "function", "function": {"name": "read"}}]

    with pytest.raises(AssertionError):
        CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)


def test_verify_accepts_far_apply_corrections_and_preserves_diagnostics():
    from onpanda import FindAndReplaceCorrectionAdapter
    from onpanda.test_utils import get_test_rejected_msgs1

    messages, far_text = get_test_rejected_msgs1()
    adapter = FindAndReplaceCorrectionAdapter(
        special_tokens={
            "split": "<|fim_pad|>",
            "stop": "<|fim_suffix|>",
            "is_good": "<|fim_prefix|>",
            "reasoning": "<|fim_middle|>",
        }
    )
    assert isinstance(adapter, FindAndReplaceCodecMixin)
    applied = adapter.apply(messages, far_text)
    correction = deepcopy(applied["correction"])
    correction.update(
        status="partial",
        fork_message_index=correction["messages_location"]["path_keys"][0],
        corrected_messages=applied["partial_messages"],
    )
    trajectory = {
        "messages": deepcopy(messages),
        "tools": None,
        "corrections": [correction],
    }

    result = CorrectionVerifier(tokenizer=utf8_tokenizer).verify(
        trajectory, trajectory
    )[0]
    assert result["find_and_replace"] == correction["find_and_replace"]
    assert result["messages_location"] == correction["messages_location"]


def test_tools_are_always_passed_to_the_native_template():
    tokenizer = BoundaryTokenizer()
    verifier = CorrectionVerifier(tokenizer=tokenizer)
    verifier._render_messages(
        [{"role": "user", "content": "Q"}],
        None,
    )
    assert tokenizer.tools_seen[-1] is None


def test_parse_failed_correction_is_scored_without_fabricating_messages():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "good", "finish_reason": "stop"},
        ],
    )
    pred = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [{"status": "parse_failed", "raw_response": "invalid"}],
    }

    rewards = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)[
        "reward_with_feedback"
    ]
    assert rewards["pred_status"] == "parse_failed"
    assert rewards["format_reward"] == 0.0
    assert rewards["final_reward"] == 0.0


def test_not_found_correction_gets_half_format_reward():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "good", "finish_reason": "stop"},
        ],
    )
    pred = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [
            {"status": "not_found", "reward_with_feedback": {"format_reward": 0.0}}
        ],
    }

    rewards = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)[
        "reward_with_feedback"
    ]
    assert rewards["pred_status"] == "not_found"
    assert rewards["format_reward"] == 0.5
    assert rewards["location_reward"] == 0.0
    assert rewards["replacement_reward"] == 0.0


def test_parse_failed_does_not_report_a_matching_fork_index_for_is_good():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "ok", "finish_reason": "stop"},
    ]
    pred = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [{"status": "parse_failed"}],
    }
    gt = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [{"fork_message_index": None}],
    }

    rewards = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)[
        "reward_with_feedback"
    ]
    assert rewards["fork_message_index_reward"] == 0.0


def test_missing_fork_index_is_parse_failed_even_with_corrected_messages():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "good", "finish_reason": "stop"},
        ],
    )
    pred = {
        "messages": deepcopy(base),
        "tools": None,
        "corrections": [
            {
                "corrected_messages": [
                    base[0],
                    {"role": "assistant", "content": "good"},
                ]
            }
        ],
    }

    rewards = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)[
        "reward_with_feedback"
    ]
    assert rewards["pred_status"] == "parse_failed"
    assert rewards["format_reward"] == 0.0
    assert rewards["final_reward"] == 0.0


def test_partial_marker_targets_the_last_empty_tool_call():
    marker = "ON_PANDA_TRUNCATE_test"
    completed_call = {"function": {"name": "first", "arguments": {}}}
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [completed_call, {}],
    }

    marked = CorrectionVerifier._append_partial_marker(message, marker)

    assert marked["tool_calls"][0] == completed_call
    assert marked["tool_calls"][1]["function"]["name"] == marker


def test_open_tool_call_sentinel_is_renderable_without_a_partial_marker():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "old", "finish_reason": "stop"},
    ]
    corrected = [
        base[0],
        {"role": "assistant", "content": "", "tool_calls": [{}]},
    ]
    trajectory = build_trajectory(base, corrected)

    rewards = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(
        trajectory, trajectory
    )["reward_with_feedback"]
    assert rewards["final_reward"] == 1.0


def test_empty_reasoning_does_not_capture_a_content_marker():
    marker = "ON_PANDA_TRUNCATE_test"
    message = {
        "role": "assistant",
        "reasoning": "",
        "content": "",
    }

    marked = CorrectionVerifier._append_partial_marker(message, marker)

    assert marked["reasoning"] == ""
    assert marked["content"] == marker


def test_nonempty_reasoning_wins_over_an_empty_duplicate_channel():
    message = {
        "role": "assistant",
        "reasoning": "think",
        "reasoning_content": "",
        "content": "answer",
    }

    prepared = CorrectionVerifier._prepare_message(message)
    assert prepared["reasoning_content"] == "think"


def test_empty_tool_call_content_is_projected_as_none_for_native_templates():
    prepared = CorrectionVerifier._prepare_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
        }
    )
    assert prepared["content"] is None


def test_unparseable_tool_arguments_use_the_placeholder_projection():
    assert (
        CorrectionVerifier._parse_partial_arguments('{"path": /tmp/b.txt}')
        == '{"ON_PANDA_PLACEHOLDER":"ON_PANDA_PLACEHOLDER"}'
    )
    prepared = CorrectionVerifier._prepare_tool_calls(
        [{"function": {"arguments": '{"path": /tmp/b.txt}'}}]
    )
    assert prepared[0]["function"]["arguments"] == {
        "ON_PANDA_PLACEHOLDER": "ON_PANDA_PLACEHOLDER"
    }


def test_unparseable_tool_arguments_do_not_match_valid_ground_truth():
    base = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read",
                        "arguments": '{"path": /tmp/b.txt}',
                    }
                }
            ],
            "finish_reason": "tool_calls",
        },
    ]
    gt = deepcopy(base)
    gt[1]["tool_calls"][0]["function"]["arguments"] = '{"path": "/tmp/a.txt"}'
    pred_message = deepcopy(base[1])
    del pred_message["finish_reason"]
    pred = build_trajectory(base, [base[0], pred_message])
    gt = build_trajectory(base, gt)

    result = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)
    assert result["reward_with_feedback"]["replacement_reward"] == 0.0


def test_partial_json_arguments_preserve_json_scalar_types():
    prepared = CorrectionVerifier._prepare_tool_calls(
        [
            {
                "function": {
                    "name": "read",
                    "arguments": '{"limit": 3, "enabled": true, "path": "/tmp/a"}',
                }
            }
        ]
    )
    arguments = prepared[0]["function"]["arguments"]
    assert arguments == {"limit": 3, "enabled": True, "path": "/tmp/a"}
    mapping = {"limit": 3, "enabled": True}
    prepared_mapping = CorrectionVerifier._prepare_tool_calls(
        [{"function": {"arguments": mapping}}]
    )
    assert prepared_mapping[0]["function"]["arguments"] == mapping


def test_text_prefix_with_different_bpe_tokens_is_not_rewarded():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "=read", "finish_reason": "stop"}],
    )
    pred = build_trajectory(base, [base[0], {"role": "assistant", "content": "="}])

    rewards = CorrectionVerifier(
        tokenizer=BoundaryTokenizer(), max_replacement_tokens=2
    ).verify_one(pred, gt)["reward_with_feedback"]
    assert rewards["location_reward"] == 1.0
    assert rewards["replacement_reward"] == 0.0


def test_text_prefix_with_different_bpe_tokens_does_not_reuse_marker_boundary():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "abc", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "abcdef", "finish_reason": "stop"},
        ],
    )
    pred = build_trajectory(base, [base[0], {"role": "assistant", "content": "abcde"}])

    rewards = CorrectionVerifier(
        tokenizer=BoundaryTokenizer(), max_replacement_tokens=20
    ).verify_one(pred, gt)["reward_with_feedback"]
    assert rewards["location_reward"] == 1.0
    assert rewards["replacement_reward"] == 0.0


def test_long_partial_prediction_matches_truncated_gt_tokens():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {
                "role": "assistant",
                "content": "abcdefghi",
                "finish_reason": "stop",
            },
        ],
    )
    pred = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "abcdefXYZ"}],
    )

    rewards = CorrectionVerifier(
        tokenizer=BoundaryTokenizer(), max_replacement_tokens=1
    ).verify_one(pred, gt)["reward_with_feedback"]
    assert rewards["replacement_reward"] == 1.0


def test_partial_prediction_shorter_than_truncated_gt_gets_no_replacement_reward():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "abcdef", "finish_reason": "stop"}],
    )
    pred = build_trajectory(base, [base[0], {"role": "assistant", "content": "a"}])

    rewards = CorrectionVerifier(
        tokenizer=utf8_tokenizer, max_replacement_tokens=2
    ).verify_one(pred, gt)["reward_with_feedback"]
    assert rewards["replacement_reward"] == 0.0


def test_partial_prediction_cannot_extend_beyond_a_short_complete_gt():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "good", "finish_reason": "stop"},
        ],
    )
    pred = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "goodx"}],
    )

    rewards = CorrectionVerifier(
        tokenizer=utf8_tokenizer, max_replacement_tokens=100
    ).verify_one(pred, gt)["reward_with_feedback"]
    assert rewards["location_reward"] == 1.0
    assert rewards["replacement_reward"] == 0.0


def test_partial_gt_can_be_a_prefix_of_a_longer_prediction_under_the_limit():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "good"}],
        status="partial",
    )
    pred = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "goodx"}],
    )

    rewards = CorrectionVerifier(
        tokenizer=utf8_tokenizer, max_replacement_tokens=100
    ).verify_one(pred, gt)["reward_with_feedback"]
    assert rewards["replacement_reward"] == 1.0


def test_complete_replacement_compares_the_truncated_gt_prefix():
    base = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "bad", "finish_reason": "stop"},
    ]
    gt = build_trajectory(
        base,
        [
            base[0],
            {"role": "assistant", "content": "abcdef", "finish_reason": "stop"},
        ],
    )
    pred = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "abX", "finish_reason": "stop"}],
    )
    wrong_pred = build_trajectory(
        base,
        [base[0], {"role": "assistant", "content": "aX", "finish_reason": "stop"}],
    )
    verifier = CorrectionVerifier(tokenizer=utf8_tokenizer, max_replacement_tokens=2)

    assert (
        verifier.verify_one(pred, gt)["reward_with_feedback"]["replacement_reward"]
        == 1.0
    )
    assert (
        verifier.verify_one(wrong_pred, gt)["reward_with_feedback"][
            "replacement_reward"
        ]
        == 0.0
    )


def test_mapping_tool_arguments_are_preserved_for_partial_rendering():
    base = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read", "arguments": {"path": "/tmp/b"}}}
            ],
            "finish_reason": "tool_calls",
        },
    ]
    gt_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "read",
                    "arguments": {"path": "/tmp/a", "limit": 1},
                }
            }
        ],
        "finish_reason": "tool_calls",
    }
    pred_message = deepcopy(gt_message)
    del pred_message["finish_reason"]
    pred = build_trajectory(base, [base[0], pred_message])
    gt = build_trajectory(base, [base[0], gt_message])

    rewards = CorrectionVerifier(tokenizer=utf8_tokenizer).verify_one(pred, gt)[
        "reward_with_feedback"
    ]
    assert rewards["final_reward"] == 1.0
