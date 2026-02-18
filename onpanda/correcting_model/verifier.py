#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 03:15:00 2026

@author: DIYer22
"""
from copy import deepcopy


class FindAndReplaceVerifier:
    default_special_tokens = dict(
        split="<|split|>",
        stop="<|stop|>",
        is_good="<|is_good|>",
        reasoning="<|reasoning|>",
    )

    def __init__(self, special_tokens=None, tokenizer=None):
        self.tokenizer = tokenizer
        self.special_tokens = {**self.default_special_tokens, **(special_tokens or {})}

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
        has_prefix = far_text.startswith(self.special_tokens["split"])
        has_suffix = far_text.endswith(self.special_tokens["split"])
        if not (has_prefix and has_suffix):
            return dict(
                parse_reward=0.0,
                parse_feedback="parse failed: missing split boundary",
                find_and_replace=default_find_and_replace,
            )

        mid_text = far_text.removeprefix(self.special_tokens["split"]).removesuffix(
            self.special_tokens["split"]
        )
        # Keep compatibility with old correcting model step1f behavior.
        if mid_text == self.special_tokens["is_good"] or mid_text in [
            "",
            self.special_tokens["split"],
        ]:
            return dict(
                parse_reward=0.5,
                parse_feedback="parse success: is_good format",
                find_and_replace=dict(
                    is_good=True,
                    location_text="",
                    location_index=0,
                    replacement_token="",
                ),
            )

        splits = mid_text.split(self.special_tokens["split"])
        if len(splits) != 3:
            return dict(
                parse_reward=0.0,
                parse_feedback=(
                    "parse failed: expect 3 fields split by "
                    f"`{self.special_tokens['split']}`"
                ),
                find_and_replace=default_find_and_replace,
            )

        location_text = splits[0]
        location_index_text = splits[1]
        replacement_token = self.special_tokens["split"].join(splits[2:])
        try:
            location_index = int(location_index_text)
        except ValueError:
            return dict(
                parse_reward=0.0,
                parse_feedback=f"parse failed: location_index must be int, got `{location_index_text}`",
                find_and_replace=default_find_and_replace,
            )

        return dict(
            parse_reward=0.5,
            parse_feedback="parse success",
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
            return dict(
                not_found=True,
                is_good=True,
                match_num=0,
                find_feedback="is_good: skip find",
            )

        location_text = find_and_replace.get("location_text", "")
        if not location_text:
            return dict(
                not_found=True,
                match_num=0,
                find_feedback="empty location_text",
            )

        location_index = find_and_replace.get("location_index")
        if not isinstance(location_index, int):
            return dict(
                not_found=True,
                match_num=0,
                find_feedback=f"location_index is not int: {location_index}",
            )

        messages_locations = []
        for path_keys, text in self._iter_assistant_text_locations(messages):
            search_scope = text + self.special_tokens["stop"]
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
            messages_location["find_feedback"] = "matched"
            return messages_location
        if match_num == 0:
            find_feedback = "location_text not found"
        else:
            find_feedback = f"location_index out of range: index={location_index}, match_num={match_num}"
        return dict(
            not_found=True,
            match_num=match_num,
            find_feedback=find_feedback,
        )

    def compute_reward(self, messages, answer_text, gt):
        assert isinstance(messages, list), type(messages)
        if messages and messages[-1]["role"] == "assistant":
            last_content = self._content_to_text(messages[-1].get("content", ""))
            assert self.special_tokens["split"] not in last_content, (
                "messages should not include FAR answer, "
                "pass FAR output via `answer_text`"
            )
        parse_res = self.parse(answer_text)

        parse_reward = parse_res["parse_reward"]
        parse_feedback = parse_res["parse_feedback"]
        pred_find_and_replace = parse_res["find_and_replace"]
        gt_find_and_replace = gt["find_and_replace"]
        gt_messages_location = gt["messages_location"]
        pred_location = self.locate(messages, pred_find_and_replace)
        match_num = pred_location.get("match_num", 0)
        find_feedback = pred_location.get("find_feedback", "")

        if pred_find_and_replace.get("is_good"):
            valid_reward = 1.0 if parse_reward > 0.0 else 0.0
            valid_feedback = "valid success: is_good format"
        elif parse_reward == 0.0:
            valid_reward = 0.0
            valid_feedback = f"valid failed: {parse_feedback}"
        else:
            find_valid_reward = 0.5 if not pred_location.get("not_found") else 0.0
            valid_reward = parse_reward + find_valid_reward
            if find_valid_reward:
                valid_feedback = f"valid success: find matched (match_num={match_num})"
            else:
                valid_feedback = (
                    "valid half: find not matched "
                    f"(reason={find_feedback}, match_num={match_num})"
                )

        gt_is_good = bool(
            gt_find_and_replace.get("is_good") or gt_messages_location.get("is_good")
        )
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
                location_feedback = f"location matched (match_num={match_num})"
            else:
                if pred_location.get("not_found"):
                    location_feedback = (
                        "location mismatch: find not matched "
                        f"(reason={find_feedback}, match_num={match_num}, "
                        f"location_text=`{pred_find_and_replace.get('location_text', '')}`), "
                        f"gt={gt_messages_location.get('path_keys')}@{gt_messages_location.get('char_index')}"
                    )
                else:
                    location_feedback = (
                        "location mismatch: "
                        f"pred={pred_location.get('path_keys')}@{pred_location.get('char_index')}, "
                        f"gt={gt_messages_location.get('path_keys')}@{gt_messages_location.get('char_index')}, "
                        f"match_num={match_num}"
                    )

            gt_replacement_token = gt_find_and_replace.get("replacement_token", "")
            pred_replacement_token = pred_find_and_replace.get("replacement_token", "")
            if not is_same_location:
                replacement_reward = 0.0
                replacement_feedback = "replacement skipped: location mismatch"
            else:
                is_same_replacement = pred_replacement_token == gt_replacement_token
                replacement_reward = 1.0 if is_same_replacement else 0.0
                if is_same_replacement:
                    replacement_feedback = "replacement matched"
                else:
                    replacement_feedback = (
                        "replacement mismatch: "
                        f"pred=`{pred_replacement_token}` gt=`{gt_replacement_token}`"
                    )

        reward = self._mean([valid_reward, location_reward, replacement_reward])
        feedback = "\n".join(
            [
                f"reward={reward:.3f}",
                f"valid_reward={valid_reward:.3f}",
                f"location_reward={location_reward:.3f}",
                f"replacement_reward={replacement_reward:.3f}",
                f"valid_feedback: {valid_feedback}",
                f"location_feedback: {location_feedback}",
                f"replacement_feedback: {replacement_feedback}",
            ]
        )
        return dict(
            reward=reward,
            valid_reward=valid_reward,
            location_reward=location_reward,
            replacement_reward=replacement_reward,
            valid_feedback=valid_feedback,
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

    split = verifier.special_tokens["split"]
    is_good = verifier.special_tokens["is_good"]
    location_text = gt["find_and_replace"]["location_text"]
    location_index = gt["find_and_replace"]["location_index"]
    replacement_token = gt["find_and_replace"]["replacement_token"]

    far_text_cases = [
        (
            "case1_all_correct",
            far_text_gt,
            dict(
                valid_reward=1.0,
                location_reward=1.0,
                replacement_reward=1.0,
            ),
        ),
        (
            "case2_wrong_replacement",
            f"{split}{location_text}{split}{location_index}{split} banana{split}",
            dict(
                valid_reward=1.0,
                location_reward=1.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case3_wrong_location_index",
            f"{split}{location_text}{split}{location_index + 1}{split}{replacement_token}{split}",
            dict(
                valid_reward=1.0,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case4_is_good_prediction",
            f"{split}{is_good}{split}",
            dict(
                valid_reward=1.0,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case5_bad_format_missing_end_split",
            f"{split}{location_text}{split}{location_index}{split}{replacement_token}",
            dict(
                valid_reward=0.0,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case6_parse_success_but_locate_not_found",
            f"{split} no_such_text{split}0{split}{replacement_token}{split}",
            dict(
                valid_reward=0.5,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
    ]

    for case_name, far_text, expected in far_text_cases:
        reward_res = verifier.compute_reward(rejected_msgs1, far_text, gt=gt)
        assert all(
            [reward_res[k] == expected[k] for k in expected]
        ), f"{case_name}, {far_text}, {expected}\n\n{reward_res}"
