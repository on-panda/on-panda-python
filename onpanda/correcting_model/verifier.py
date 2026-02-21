#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 03:15:00 2026

@author: DIYer22
"""


class FindAndReplaceVerifier:
    """
    Tokenizer-agnostic FindAndReplaceVerifier
    This class can standalone without dependency.
    """

    default_special_tokens = dict(
        split="<|split|>",
        stop="<|stop|>",
        is_good="<|is_good|>",
        reasoning="<|reasoning|>",
    )

    def __init__(self, special_tokens=None):
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

    def _normalize_replacement_token(self, replacement_token):
        """
        Normalize the replacement_token by removing stop token, and return whether it has the stop token.
        """
        stop_token = self.special_tokens["stop"]
        has_stop_token = stop_token in replacement_token
        normalized_replacement_token = replacement_token.split(stop_token, 1)[0]
        return normalized_replacement_token, has_stop_token

    def _build_good_prefix_and_char_location(self, rejected_content_str, correction):
        path_keys = correction["messages_location"]["path_keys"]
        char_index = correction["messages_location"]["char_index"]
        replacement_token = correction["find_and_replace"]["replacement_token"]
        normalized_replacement_token, _ = self._normalize_replacement_token(
            replacement_token
        )
        good_prefix = rejected_content_str[:char_index] + normalized_replacement_token
        fork_char_index = char_index
        max_common_len = min(len(rejected_content_str), len(good_prefix))
        while (
            fork_char_index < max_common_len
            and rejected_content_str[fork_char_index] == good_prefix[fork_char_index]
        ):
            fork_char_index += 1
        char_messages_location = dict(
            path_keys=list(path_keys), char_index=fork_char_index
        )
        return good_prefix, char_messages_location

    def parse(self, far_text):
        default_find_and_replace = dict(
            is_good=False,
            location_text="",
            location_index=0,
            replacement_token="",
            far_text=far_text,
        )
        has_prefix = far_text.startswith(self.special_tokens["split"])
        has_suffix = far_text.endswith(self.special_tokens["split"])
        if not (has_prefix and has_suffix):
            return dict(
                reward_with_feedback=dict(
                    parse_reward=0.0,
                    parse_feedback="parse failed: missing split boundary",
                ),
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
                reward_with_feedback=dict(
                    parse_reward=0.5, parse_feedback="parse success: is_good format"
                ),
                find_and_replace=dict(
                    is_good=True,
                    location_text="",
                    location_index=0,
                    replacement_token="",
                    far_text=far_text,
                ),
            )

        splits = mid_text.split(self.special_tokens["split"])
        if len(splits) != 3:
            return dict(
                reward_with_feedback=dict(
                    parse_reward=0.0,
                    parse_feedback=(
                        "parse failed: expect 3 fields split by "
                        f"`{self.special_tokens['split']}`"
                    ),
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
                reward_with_feedback=dict(
                    parse_reward=0.0,
                    parse_feedback=f"parse failed: location_index must be int, got `{location_index_text}`",
                ),
                find_and_replace=default_find_and_replace,
                far_text=far_text,
            )
        too_long_reason = ""
        MAX_SINGLE_TOKEN_LEN = 128
        if len(location_text) > 21 * MAX_SINGLE_TOKEN_LEN:
            too_long_reason = "location_text is too long"
        elif len(replacement_token) > 4 * MAX_SINGLE_TOKEN_LEN:
            too_long_reason = "replacement_token is too long"
        if too_long_reason:
            return dict(
                reward_with_feedback=dict(
                    parse_reward=0.0,
                    parse_feedback=f"parse failed: {too_long_reason}",
                ),
                find_and_replace=default_find_and_replace,
            )

        return dict(
            reward_with_feedback=dict(parse_reward=0.5, parse_feedback="parse success"),
            find_and_replace=dict(
                is_good=False,
                location_text=location_text,
                location_index=location_index,
                replacement_token=replacement_token,
                far_text=far_text,
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

    def assert_messages_location_context_valid(self, messages, messages_location):
        if "left5" not in messages_location:
            return

        path_keys = list(messages_location["path_keys"])
        char_index = messages_location["char_index"]
        text = messages
        for key in path_keys:
            text = text[key]
        text = self._content_to_text(text)
        left5 = messages_location["left5"]
        right5 = messages_location["right5"]
        assert left5 == text[max(0, char_index - 5) : char_index], (
            "messages_location.left5 is invalid for current messages: "
            f"left5={left5!r}, path_keys={path_keys}, char_index={char_index}"
        )
        assert right5 == text[char_index : char_index + 5], (
            "messages_location.right5 is invalid for current messages: "
            f"right5={right5!r}, path_keys={path_keys}, char_index={char_index}"
        )

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
                    left5=text[max(0, index - 5) : index],
                    right5=text[index : index + 5],
                )
                messages_locations.append(messages_location)
                start = index + 1
        match_num = len(messages_locations)
        if match_num and -match_num <= location_index < match_num:
            messages_location = __import__("copy").deepcopy(
                messages_locations[location_index]
            )
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

    def parse_and_locate(self, messages, far_text):
        parse_res = self.parse(far_text)
        find_and_replace = parse_res["find_and_replace"]
        messages_location = self.locate(messages, find_and_replace)
        return dict(
            find_and_replace=find_and_replace,
            messages_location=messages_location,
            reward_with_feedback=parse_res["reward_with_feedback"],
        )

    def compute_reward(self, messages, far_text, gt_correction):
        """
        reward computation logic:
        ```python
        F = format_reward
        L = location
        R = replacement

        if parse_fail:
            return F0,L0,R0
        if not_found:
            return F0.5,L0,R0
        _build_good_prefix_and_char_fork_location for both pred and gt
        if pred_location != gt_location:
            return F1,L0,R0
        if pred_good_prefix.startswith(gt_good_prefix):
            return F1,L1,R1
        return F1,L1,R0
        ```
        """
        assert isinstance(messages, list), type(messages)
        if messages and messages[-1]["role"] == "assistant":
            last_content = self._content_to_text(messages[-1].get("content", ""))
            assert self.special_tokens["split"] not in last_content, (
                "messages should not include FAR answer, "
                "pass FAR output via `far_text`"
            )
        correction = self.parse_and_locate(messages, far_text)
        reward_with_feedback = correction.pop("reward_with_feedback")
        parse_reward = reward_with_feedback["parse_reward"]
        parse_feedback = reward_with_feedback["parse_feedback"]
        pred_find_and_replace = correction["find_and_replace"]
        gt_find_and_replace = gt_correction["find_and_replace"]
        gt_messages_location = gt_correction["messages_location"]
        pred_location = correction["messages_location"]
        match_num = pred_location.get("match_num", 0)
        find_feedback = pred_location.get("find_feedback", "")

        # Compute format_reward
        if pred_find_and_replace.get("is_good"):
            format_reward = 1.0 if parse_reward > 0.0 else 0.0
            format_feedback = "format success: is_good format"
        elif parse_reward == 0.0:
            format_reward = 0.0
            format_feedback = f'format failed: "{parse_feedback}"'
        else:
            find_format_reward = 0.5 if not pred_location.get("not_found") else 0.0
            format_reward = parse_reward + find_format_reward
            if find_format_reward:
                format_feedback = (
                    f"format success: find matched (match_num={match_num})"
                )
            else:
                format_feedback = (
                    "format half: find not matched "
                    f'(reason="{find_feedback}", match_num={match_num})'
                )

        gt_is_good = bool(
            gt_find_and_replace.get("is_good") or gt_messages_location.get("is_good")
        )
        if gt_is_good:
            location_reward = 1.0 if pred_find_and_replace.get("is_good") else 0.0
            replacement_reward = location_reward
            location_feedback = (
                "location skipped: gt is_good"
                if location_reward
                else "location mismatch: gt is_good"
            )
            replacement_feedback = (
                "replacement skipped: gt is_good"
                if replacement_reward
                else "replacement mismatch: gt is_good"
            )
        else:
            if "left5" in gt_messages_location:
                self.assert_messages_location_context_valid(
                    messages,
                    gt_messages_location,
                )
            pred_is_good = bool(
                pred_find_and_replace.get("is_good") or pred_location.get("is_good")
            )
            if pred_is_good:
                location_reward = 0.0
                replacement_reward = 0.0
                location_feedback = "location mismatch: pred is_good"
                replacement_feedback = "replacement mismatch: pred is_good"
            else:
                gt_path_keys = gt_messages_location["path_keys"]

                pred_not_found = pred_location.get("not_found")
                pred_path_keys = pred_location.get("path_keys")
                if pred_not_found or pred_path_keys != gt_path_keys:
                    location_reward = 0.0
                    replacement_reward = 0.0
                    if pred_not_found:
                        location_feedback = "location mismatch: find not matched"
                        replacement_feedback = "replacement skipped: find not matched"
                    else:
                        location_feedback = "location mismatch: path_keys mismatch"
                        replacement_feedback = "replacement skipped: path_keys mismatch"
                else:  # using tokenizer_agnostic_loose_reward for location_reward and replacement_reward
                    gt_content = messages
                    for key in gt_path_keys:
                        gt_content = gt_content[key]
                    rejected_content_str = self._content_to_text(gt_content)

                    # using tokenizer-agnostic char level, instead of token level, for fork location index in verifier
                    gt_good_prefix, gt_char_messages_location = (
                        self._build_good_prefix_and_char_location(
                            rejected_content_str=rejected_content_str,
                            correction=gt_correction,
                        )
                    )
                    pred_good_prefix, pred_char_messages_location = (
                        self._build_good_prefix_and_char_location(
                            rejected_content_str=rejected_content_str,
                            correction=correction,
                        )
                    )
                    # tokenizer_agnostic location reward
                    location_reward = float(
                        self._has_same_location(
                            pred_char_messages_location,
                            gt_char_messages_location,
                        )
                    )
                    if location_reward:
                        location_feedback = "location matched: fork"
                        _, gt_has_stop_token = self._normalize_replacement_token(
                            gt_find_and_replace["replacement_token"]
                        )
                        _, pred_has_stop_token = self._normalize_replacement_token(
                            pred_find_and_replace["replacement_token"]
                        )
                        if gt_has_stop_token:
                            replacement_reward = float(
                                pred_has_stop_token
                                and pred_good_prefix == gt_good_prefix
                            )
                        else:
                            # loose_reward that allow longer replacement_tokens
                            replacement_reward = float(
                                pred_good_prefix.startswith(gt_good_prefix)
                            )
                    else:
                        location_feedback = "location mismatch: fork"
                        replacement_reward = 0.0

                    if replacement_reward:
                        replacement_feedback = "replacement matched: prefix"
                    else:
                        replacement_feedback = "replacement mismatch: prefix"

        final_reward = self._mean([format_reward, location_reward, replacement_reward])
        feedback = "\n".join(
            [
                f"final_reward = {round(final_reward, 3)}",
                f"format_reward = {round(format_reward, 3)}",
                f"location_reward = {round(location_reward, 3)}",
                f"replacement_reward = {round(replacement_reward, 3)}",
                f"format_feedback: {format_feedback}",
                f"location_feedback: {location_feedback}",
                f"replacement_feedback: {replacement_feedback}",
                "Note: Each reward ∈ [0, 1], and final_reward is the average of the other rewards.",
            ]
        )
        reward_with_feedback = dict(
            final_reward=final_reward,
            format_reward=format_reward,
            location_reward=location_reward,
            replacement_reward=replacement_reward,
            format_feedback=format_feedback,
            feedback=feedback,
        )
        reward_result = dict(reward_with_feedback=reward_with_feedback, **correction)
        return reward_result


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
    gt_correction = adapter.build_correction_from_rejected_messages(rejected_msgs1)

    split = verifier.special_tokens["split"]
    is_good = verifier.special_tokens["is_good"]
    location_text = gt_correction["find_and_replace"]["location_text"]
    location_index = gt_correction["find_and_replace"]["location_index"]
    replacement_token = gt_correction["find_and_replace"]["replacement_token"]

    far_text_cases = [
        (
            "case1_all_correct",
            far_text_gt,
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=1.0,
            ),
        ),
        (
            "case2_wrong_replacement",
            f"{split}{location_text}{split}{location_index}{split} banana{split}",
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case3_wrong_location_index",
            f"{split}{location_text}{split}{location_index + 1}{split}{replacement_token}{split}",
            dict(
                format_reward=1.0,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case4_is_good_prediction",
            f"{split}{is_good}{split}",
            dict(
                format_reward=1.0,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case5_bad_format_missing_end_split",
            f"{split}{location_text}{split}{location_index}{split}{replacement_token}",
            dict(
                format_reward=0.0,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case6_parse_success_but_locate_not_found",
            f"{split} no_such_text{split}0{split}{replacement_token}{split}",
            dict(
                format_reward=0.5,
                location_reward=0.0,
                replacement_reward=0.0,
            ),
        ),
        (
            "case7_tokenizer_agnostic_loose_reward",
            f"{split}potato, banana{split}{location_index}{split}orange, pineapple{split}",
            dict(
                format_reward=1.0,
                location_reward=1.0,
                replacement_reward=1.0,
            ),
        ),
    ]

    for case_name, far_text, expected in far_text_cases:
        result = verifier.compute_reward(
            rejected_msgs1, far_text, gt_correction=gt_correction
        )
        reward_res = result["reward_with_feedback"]
        assert all(
            [reward_res[k] == expected[k] for k in expected]
        ), f"{case_name}, {far_text}, {expected}\n\n{reward_res}"
        print(f"\n\n{case_name} passed: \n{far_text}\n{reward_res['feedback']}")
