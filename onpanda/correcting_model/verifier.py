#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 03:15:00 2026

@author: DIYer22
"""
from copy import deepcopy


class FindAndReplaceVerifier:
    def __init__(self, tokenizer, special_tokens):
        self.tokenizer = tokenizer
        special_tokens = special_tokens or {}
        self.SPLIT_TOKEN = special_tokens.get("split", "<|split|>")
        self.STOP_TOKEN = special_tokens.get("stop", "<|stop|>")
        self.IS_GOOD_TOKEN = special_tokens.get("is_good", "<|is_good|>")
        self.REASONING_TOKEN = special_tokens.get("reasoning", "<|reasoning|>")

    @staticmethod
    def _mean(nums):
        return sum(nums) / len(nums)

    @staticmethod
    def _content_to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            assert all([d["type"] == "text" for d in content]), content
            return "".join([d.get("text", "") for d in content])
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)

    @staticmethod
    def _has_same_location(location1, location2):
        return (
            isinstance(location1, dict)
            and isinstance(location2, dict)
            and location1.get("path_keys") == location2.get("path_keys")
            and location1.get("char_index") == location2.get("char_index")
        )

    def parse(self, far_text):
        default_find_and_replace = dict(
            is_good=False,
            location_text="",
            location_index=0,
            replacement_token="",
        )
        if not isinstance(far_text, str):
            return dict(
                parse_reward=0.0,
                parse_feedback=f"far_text must be str, got {type(far_text).__name__}",
                find_and_replace=default_find_and_replace,
            )

        parse_reward = 0.0
        parse_feedbacks = []

        has_prefix = far_text.startswith(self.SPLIT_TOKEN)
        has_suffix = far_text.endswith(self.SPLIT_TOKEN)
        if has_prefix:
            parse_reward += 0.25
        else:
            parse_feedbacks.append("missing start split token")
        if has_suffix:
            parse_reward += 0.25
        else:
            parse_feedbacks.append("missing end split token")
        if not (has_prefix and has_suffix):
            return dict(
                parse_reward=parse_reward,
                parse_feedback="; ".join(parse_feedbacks),
                find_and_replace=default_find_and_replace,
            )

        mid_text = far_text.removeprefix(self.SPLIT_TOKEN).removesuffix(
            self.SPLIT_TOKEN
        )
        # Keep compatibility with old correcting model behavior.
        if mid_text == self.IS_GOOD_TOKEN or mid_text in ["", self.SPLIT_TOKEN]:
            return dict(
                parse_reward=1.0,
                parse_feedback="parse success: is_good format",
                find_and_replace=dict(
                    is_good=True,
                    location_text="",
                    location_index=0,
                    replacement_token="",
                ),
            )

        splits = mid_text.split(self.SPLIT_TOKEN)
        if len(splits) >= 3:
            parse_reward += 0.25
            if len(splits) > 3:
                parse_reward -= 0.1
                parse_feedbacks.append(
                    "found extra split token in replacement, merged as tail text"
                )
        else:
            parse_feedbacks.append(
                f"expect 3 fields split by `{self.SPLIT_TOKEN}`, got {len(splits)}"
            )
            return dict(
                parse_reward=parse_reward,
                parse_feedback="; ".join(parse_feedbacks),
                find_and_replace=default_find_and_replace,
            )

        location_text = splits[0]
        location_index_text = splits[1]
        replacement_token = self.SPLIT_TOKEN.join(splits[2:])
        try:
            location_index = int(location_index_text)
            parse_reward += 0.25
        except ValueError:
            location_index = 0
            parse_feedbacks.append(
                f"location_index must be int, got `{location_index_text}`"
            )

        if parse_reward >= 0.99:
            parse_feedback = "parse success"
            parse_reward = 1.0
        else:
            parse_feedback = "; ".join(parse_feedbacks)

        return dict(
            parse_reward=parse_reward,
            parse_feedback=parse_feedback,
            find_and_replace=dict(
                is_good=False,
                location_text=location_text,
                location_index=location_index,
                replacement_token=replacement_token,
            ),
        )

    def _iter_assistant_text_locations(self, messages):
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str):
                yield [message_index, "reasoning"], reasoning

            content = self._content_to_text(message.get("content", ""))
            yield [message_index, "content"], content

            for tool_call_index, tool_call in enumerate(message.get("tool_calls", [])):
                function = tool_call.get("function", {})
                function_name = function.get("name")
                if isinstance(function_name, str):
                    yield [
                        message_index,
                        "tool_calls",
                        tool_call_index,
                        "function",
                        "name",
                    ], function_name
                function_arguments = function.get("arguments")
                if isinstance(function_arguments, str):
                    yield [
                        message_index,
                        "tool_calls",
                        tool_call_index,
                        "function",
                        "arguments",
                    ], function_arguments

    def locate(self, messages, find_and_replace):
        if find_and_replace.get("is_good"):
            return dict(not_found=True, is_good=True)

        location_text = find_and_replace.get("location_text", "")
        if not location_text:
            return dict(not_found=True, match_num=0)

        location_index = find_and_replace.get("location_index")
        if not isinstance(location_index, int):
            return dict(not_found=True, match_num=0)

        messages_locations = []
        for path_keys, text in self._iter_assistant_text_locations(messages):
            search_scope = text + self.STOP_TOKEN
            start = 0
            while True:
                index = search_scope.find(location_text, start)
                if index == -1:
                    break
                messages_location = dict(
                    path_keys=path_keys,
                    char_index=index,
                    left5=search_scope[max(0, index - 5) : index],
                    right5=search_scope[index : index + 5],
                )
                messages_locations.append(messages_location)
                start = index + 1
        match_num = len(messages_locations)
        if match_num and -match_num <= location_index < match_num:
            messages_location = deepcopy(messages_locations[location_index])
            messages_location["match_num"] = match_num
            messages_location["patch_length"] = len(location_text)
            return messages_location
        return dict(not_found=True, match_num=match_num)

    def compute_reward(self, messages, answer_text, gt):
        assert isinstance(messages, list), type(messages)
        if messages and messages[-1]["role"] == "assistant":
            last_content = self._content_to_text(messages[-1].get("content", ""))
            assert self.SPLIT_TOKEN not in last_content, (
                "messages should not include FAR answer, "
                "pass FAR output via `answer_text`"
            )
        parse_res = self.parse(answer_text)

        parse_reward = parse_res["parse_reward"]
        parse_feedback = parse_res["parse_feedback"]
        pred_find_and_replace = parse_res["find_and_replace"]
        gt_find_and_replace = gt["find_and_replace"]
        gt_messages_location = gt["messages_location"]

        gt_is_good = bool(
            gt_find_and_replace.get("is_good") or gt_messages_location.get("is_good")
        )
        pred_location = self.locate(messages, pred_find_and_replace)
        if gt_is_good:
            location_reward = 1.0 if pred_find_and_replace.get("is_good") else 0.0
            replacement_reward = location_reward
            location_feedback = (
                "location skipped: ground truth is_good"
                if location_reward
                else "ground truth is_good but prediction is not is_good"
            )
            replacement_feedback = (
                "replacement skipped: ground truth is_good"
                if replacement_reward
                else "ground truth is_good but prediction is not is_good"
            )
        else:
            is_same_location = self._has_same_location(
                pred_location,
                gt_messages_location,
            )
            location_reward = 1.0 if is_same_location else 0.0
            if is_same_location:
                location_feedback = "location matched"
            else:
                location_feedback = (
                    "location mismatch: "
                    f"pred={pred_location.get('path_keys')}@{pred_location.get('char_index')}, "
                    f"gt={gt_messages_location.get('path_keys')}@{gt_messages_location.get('char_index')}"
                )

            gt_replacement_token = gt_find_and_replace.get("replacement_token", "")
            pred_replacement_token = pred_find_and_replace.get("replacement_token", "")
            is_same_replacement = pred_replacement_token == gt_replacement_token
            replacement_reward = 1.0 if is_same_replacement else 0.0
            if is_same_replacement:
                replacement_feedback = "replacement matched"
            else:
                replacement_feedback = (
                    "replacement mismatch: "
                    f"pred=`{pred_replacement_token}` gt=`{gt_replacement_token}`"
                )

        reward = self._mean([parse_reward, location_reward, replacement_reward])
        feedback = "\n".join(
            [
                f"reward={reward:.3f}",
                f"parse_reward={parse_reward:.3f}",
                f"location_reward={location_reward:.3f}",
                f"replacement_reward={replacement_reward:.3f}",
                f"parse_feedback: {parse_feedback}",
                f"location_feedback: {location_feedback}",
                f"replacement_feedback: {replacement_feedback}",
            ]
        )
        return dict(
            reward=reward,
            parse_reward=parse_reward,
            location_reward=location_reward,
            replacement_reward=replacement_reward,
            parse_feedback=parse_feedback,
            feedback=feedback,
            pred_find_and_replace=pred_find_and_replace,
            pred_messages_location=pred_location,
        )


if __name__ == "__main__":
    import os
    import sys
    import onpanda
    from onpanda.test_utils import get_test_rejected_msgs1

    rejected_msgs1, far_text_gt = get_test_rejected_msgs1()
    adapter = onpanda.FindAndReplaceCorrectionAdapter(
        special_tokens=dict(
            split="<|fim_pad|>",
            stop="<|fim_suffix|>",
            is_good="<|fim_prefix|>",
            reasoning="<|fim_middle|>",
        )
    )
    verifier = adapter.verifier
    gt = adapter.build_correction_from_rejected_messages(rejected_msgs1)

    split = verifier.SPLIT_TOKEN
    is_good = verifier.IS_GOOD_TOKEN
    location_text = gt["find_and_replace"]["location_text"]
    location_index = gt["find_and_replace"]["location_index"]
    replacement_token = gt["find_and_replace"]["replacement_token"]

    far_text_cases = [
        (
            "case1_all_correct",
            far_text_gt,
            dict(parse_reward=1.0, location_reward=1.0, replacement_reward=1.0),
        ),
        (
            "case2_wrong_replacement",
            f"{split}{location_text}{split}{location_index}{split} banana{split}",
            dict(parse_reward=1.0, location_reward=1.0, replacement_reward=0.0),
        ),
        (
            "case3_wrong_location_index",
            f"{split}{location_text}{split}{location_index + 1}{split}{replacement_token}{split}",
            dict(parse_reward=1.0, location_reward=0.0, replacement_reward=1.0),
        ),
        (
            "case4_is_good_prediction",
            f"{split}{is_good}{split}",
            dict(parse_reward=1.0, location_reward=0.0, replacement_reward=0.0),
        ),
        (
            "case5_bad_format_missing_end_split",
            f"{split}{location_text}{split}{location_index}{split}{replacement_token}",
            dict(parse_reward=0.25, location_reward=0.0, replacement_reward=0.0),
        ),
    ]

    for case_name, far_text, expected in far_text_cases:
        reward_res = verifier.compute_reward(rejected_msgs1, far_text, gt=gt)
        assert all(
            [reward_res[k] == expected[k] for k in expected]
        ), f"{case_name}, {far_text}, {expected}\n\n{reward_res}"
