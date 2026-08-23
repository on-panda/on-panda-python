#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 03:15:00 2026

@author: DIYer22
"""

import json
import uuid
from copy import deepcopy

import mximport

with mximport.inpkg():
    from ..response_templates import build_messages_location, build_response_template
    from ..response_templates.partial_json import parse_partial_json_object
    from ..token_level_supervision_utils import build_tokenizer

ON_PANDA_PLACEHOLDER = "ON_PANDA_PLACEHOLDER"
ON_PANDA_TRUNCATE_PREFIX = "ON_PANDA_TRUNCATE_"
ON_PANDA_INVALID_ARGUMENTS = '{"ON_PANDA_PLACEHOLDER":"ON_PANDA_PLACEHOLDER"}'


class FindAndReplaceCodecMixin:
    """
    FAR wire parsing and legacy locate/reward helpers.

    The correction verifier below compares canonical trajectories in tokenizer space and does
    not depend on this mixin.
    """

    default_special_tokens = dict(
        split="<|split|>",
        stop="<|stop|>",
        is_good="<|is_good|>",
        reasoning="<|reasoning|>",
    )

    def __init__(self, special_tokens=None, response_template=None):
        self.special_tokens = {**self.default_special_tokens, **(special_tokens or {})}
        self.response_template = build_response_template(
            response_template, special_tokens=self.special_tokens
        )

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
                    parse_reward=1.0, parse_feedback="parse success: is_good format"
                ),
                find_and_replace=dict(
                    is_good=True,
                    location_text="",
                    location_index=0,
                    replacement_token="",
                    far_text=far_text,
                ),
            )

        split_token = self.special_tokens["split"]
        split_count = far_text.count(split_token)
        if split_count != 4:
            return dict(
                reward_with_feedback=dict(
                    parse_reward=0.0,
                    parse_feedback=(
                        f"parse failed: expect count(`{split_token}`) == 4, "
                        f"got {split_count}"
                    ),
                ),
                find_and_replace=default_find_and_replace,
            )

        location_text, location_index_text, replacement_token = mid_text.split(
            split_token,
            2,
        )
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
            reward_with_feedback=dict(parse_reward=1.0, parse_feedback="parse success"),
            find_and_replace=dict(
                is_good=False,
                location_text=location_text,
                location_index=location_index,
                replacement_token=replacement_token,
                far_text=far_text,
            ),
        )

    def build_templated_location(self, messages, message_index):
        """
        Correcting happens in the response template's text space, where every channel end is
        addressable, including reasoning end and each tool call end.
        """
        apply_result = self.response_template.apply(messages[message_index])
        return dict(
            message_index=message_index,
            message=messages[message_index],
            **apply_result,
        )

    def _iter_assistant_templated_prompts(self, messages):
        for message_index, message in enumerate(messages):
            if message["role"] == "assistant":
                yield self.build_templated_location(messages, message_index)

    def build_messages_location(self, templated_location, find_and_replace):
        """messages_location of a located match, with the find feedback of that match."""
        return dict(
            build_messages_location(
                templated_location, templated_location["templated_char_index"]
            ),
            match_num=templated_location["match_num"],
            patch_length=len(find_and_replace["location_text"]),
            find_feedback="matched",
        )

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

    def locate_templated(self, messages, find_and_replace):
        """Find location_text in template space, then pick the match by location_index."""
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

        templated_locations = []
        for templated_location in self._iter_assistant_templated_prompts(messages):
            templated_prompt = templated_location["templated_prompt"]
            start = 0
            while True:
                index = templated_prompt.find(location_text, start)
                if index == -1:
                    break
                templated_locations.append(
                    dict(templated_location, templated_char_index=index)
                )
                start = index + 1
        match_num = len(templated_locations)
        if match_num and -match_num <= location_index < match_num:
            return dict(
                templated_locations[location_index],
                match_num=match_num,
                find_feedback="matched",
            )
        if match_num == 0:
            find_feedback = "location_text not found"
        else:
            find_feedback = f"location_index out of range: index={location_index}, match_num={match_num}"
        return dict(
            not_found=True,
            match_num=match_num,
            find_feedback=find_feedback,
        )

    def locate(self, messages, find_and_replace):
        if find_and_replace.get("is_good"):
            return dict(
                not_found=True,
                is_good=True,
                match_num=0,
                find_feedback="is_good: skip find",
            )
        templated_location = self.locate_templated(messages, find_and_replace)
        if templated_location.get("not_found"):
            return dict(
                not_found=True,
                match_num=templated_location["match_num"],
                find_feedback=templated_location["find_feedback"],
            )
        return self.build_messages_location(templated_location, find_and_replace)

    def parse_and_locate(self, messages, far_text):
        parse_res = self.parse(far_text)
        find_and_replace = parse_res["find_and_replace"]
        messages_location = self.locate(messages, find_and_replace)
        return dict(
            find_and_replace=find_and_replace,
            messages_location=messages_location,
            reward_with_feedback=parse_res["reward_with_feedback"],
        )

    # TODO: Remove after PandaScoreMixin and legacy FAR data migrate to trajectories.
    def compute_reward(self, messages, far_text, gt_correction):
        """Deprecated: score a legacy FAR correction for compatibility only.

        New trajectory scoring should use ``CorrectionVerifier.verify`` or
        ``CorrectionVerifier.verify_one``.

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
        gt_is_good = bool(
            gt_find_and_replace.get("is_good") or gt_messages_location.get("is_good")
        )
        pred_is_good = bool(
            pred_find_and_replace.get("is_good") or pred_location.get("is_good")
        )
        is_good_cls_reward = (
            float(gt_is_good == pred_is_good) if parse_reward > 0.0 else 0.0
        )

        # Compute format_reward
        if pred_find_and_replace.get("is_good"):
            find_reward = 1.0 if parse_reward > 0.0 else 0.0
            format_reward = self._mean([parse_reward, find_reward])
            format_feedback = "format success: is_good format"
        elif parse_reward == 0.0:
            find_reward = 0.0
            format_reward = 0.0
            format_feedback = f'format failed: "{parse_feedback}"'
        else:
            find_reward = 1.0 if not pred_location.get("not_found") else 0.0
            format_reward = self._mean([parse_reward, find_reward])
            if find_reward:
                format_feedback = (
                    f"format success: find matched (match_num={match_num})"
                )
            else:
                format_feedback = (
                    "format half: find not matched "
                    f'(reason="{find_feedback}", match_num={match_num})'
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
                else:  # legacy loose reward for location_reward and replacement_reward
                    gt_content = messages
                    for key in gt_path_keys:
                        gt_content = gt_content[key]
                    rejected_content_str = self._content_to_text(gt_content)

                    # Legacy character-level fork location, kept for the FAR API.
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
                    # Legacy character-level location reward.
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
                f"parse_reward = {round(parse_reward, 3)}",
                f"find_reward = {round(find_reward, 3)}",
                f"format_reward = {round(format_reward, 3)}",
                f"location_reward = {round(location_reward, 3)}",
                f"replacement_reward = {round(replacement_reward, 3)}",
                f"format_feedback: {format_feedback}",
                f"location_feedback: {location_feedback}",
                f"replacement_feedback: {replacement_feedback}",
                "Note: Each reward ∈ [0, 1], format_reward is avg(parse_reward, find_reward), and final_reward is avg(format_reward, location_reward, replacement_reward).",
            ]
        )
        reward_with_feedback = dict(
            final_reward=final_reward,
            parse_reward=parse_reward,
            find_reward=find_reward,
            format_reward=format_reward,
            location_reward=location_reward,
            replacement_reward=replacement_reward,
            is_good_cls_reward=is_good_cls_reward,
            format_feedback=format_feedback,
            feedback=feedback,
        )
        reward_result = dict(reward_with_feedback=reward_with_feedback, **correction)
        return reward_result


class CorrectionVerifier:
    """Compare trajectory corrections in the configured tokenizer's token space."""

    complete_finish_reasons = frozenset(("stop", "tool_calls"))
    partial_finish_reasons = frozenset(("length", "reasoning_end"))
    query_placeholder = "QUERY_PLACEHOLDER"
    placeholder = ON_PANDA_PLACEHOLDER

    def __init__(
        self,
        tokenizer=None,
        max_replacement_tokens=None,
        *,
        multi_correction_gt_mode="mean",
    ):
        self.tokenizer = build_tokenizer(tokenizer)
        if max_replacement_tokens is None:
            max_replacement_tokens = 2 if tokenizer is None else 1
        self.max_replacement_tokens = max_replacement_tokens
        assert multi_correction_gt_mode in ("max", "mean")
        self.multi_correction_gt_mode = multi_correction_gt_mode
        self._tool_arguments_type = None

    @classmethod
    def _append_marker_to_value(cls, value, marker):
        if isinstance(value, str):
            return value + marker
        if isinstance(value, list):
            value = deepcopy(value)
            for item in reversed(value):
                if isinstance(item, dict) and "text" in item:
                    item["text"] = str(item["text"]) + marker
                    return value
            value.append(dict(type="text", text=marker))
            return value
        if isinstance(value, dict) and "text" in value:
            value = deepcopy(value)
            value["text"] = str(value["text"]) + marker
            return value
        if value is None:
            return marker
        return str(value) + marker

    @classmethod
    def _append_partial_marker(cls, message, marker):
        message = deepcopy(message)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            call = tool_calls[-1]
            if call is None:
                tool_calls[-1] = {"function": {"name": marker, "arguments": {}}}
                return message
            function = call.setdefault("function", {})
            if "arguments" in function:
                arguments = function["arguments"]
                if isinstance(arguments, dict):
                    arguments = deepcopy(arguments)
                    if arguments:
                        last_name = next(reversed(arguments))
                        last_value = arguments[last_name]
                        if not isinstance(last_value, str):
                            last_value = cls._stringify_tool_value(last_value)
                        arguments[last_name] = last_value + marker
                    else:
                        arguments[marker] = cls.placeholder
                    function["arguments"] = arguments
                else:
                    function["arguments"] = cls._append_marker_to_value(
                        arguments, marker
                    )
            elif "name" in function:
                function["name"] = cls._append_marker_to_value(function["name"], marker)
            else:
                function["name"] = marker
            return message

        reasoning = message.get("reasoning")
        if not reasoning:
            reasoning = message.get("reasoning_content")
        content = message.get("content")
        if (
            message.get("finish_reason") == "reasoning_end"
            and reasoning is not None
            and (content is None or content == "")
            and not tool_calls
        ):
            message["content"] = marker
            return message
        if reasoning and (content is None or content == ""):
            marked_reasoning = cls._append_marker_to_value(reasoning, marker)
            message["reasoning"] = marked_reasoning
            message["reasoning_content"] = marked_reasoning
        elif "content" in message:
            message["content"] = cls._append_marker_to_value(content, marker)
        else:
            message["content"] = marker
        return message

    @staticmethod
    def _stringify_tool_value(value):
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _parse_partial_arguments(cls, raw_arguments, marker=None):
        if not isinstance(raw_arguments, str):
            return raw_arguments
        marker = marker or ""
        clean_arguments = raw_arguments.replace(marker, "") if marker else raw_arguments
        parsed = parse_partial_json_object(clean_arguments)
        if parsed is None:
            return ON_PANDA_INVALID_ARGUMENTS
        entries = parsed["entries"]
        arguments = {}
        for entry in entries:
            name = entry["name"]
            if not entry["name_complete"]:
                name += marker
            value = entry.get("value", cls.placeholder)
            if marker and (not entry.get("complete") or entry is entries[-1]):
                value = cls._stringify_tool_value(value) + marker
            arguments[name] = value
        if marker and not arguments:
            arguments[marker] = cls.placeholder
        return arguments

    @classmethod
    def _prepare_tool_calls(cls, tool_calls, marker=None, arguments_type="mapping"):
        if tool_calls == [{}]:
            return [
                {
                    "function": {
                        "name": marker or cls.placeholder,
                        "arguments": {},
                    }
                }
            ]
        prepared = []
        for call in tool_calls:
            call = deepcopy(call)
            function = call.setdefault("function", {})
            if "name" not in function:
                function["name"] = marker or cls.placeholder
            if "arguments" in function:
                arguments = function["arguments"]
                if isinstance(arguments, dict):
                    function["arguments"] = (
                        arguments
                        if arguments_type == "mapping"
                        else json.dumps(arguments, ensure_ascii=False)
                    )
                else:
                    if arguments_type == "string":
                        if isinstance(arguments, str):
                            clean_arguments = (
                                arguments.replace(marker, "") if marker else arguments
                            )
                            if parse_partial_json_object(clean_arguments) is None:
                                arguments = ON_PANDA_INVALID_ARGUMENTS + (marker or "")
                        else:
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        function["arguments"] = arguments
                    else:
                        parsed_arguments = cls._parse_partial_arguments(
                            arguments, marker=marker
                        )
                        if parsed_arguments == ON_PANDA_INVALID_ARGUMENTS:
                            parsed_arguments = {
                                cls.placeholder: marker or cls.placeholder
                            }
                        function["arguments"] = parsed_arguments
            elif marker:
                function["arguments"] = {cls.placeholder: cls.placeholder}
            ordered_function = {}
            for key in ("name", "arguments"):
                if key in function:
                    ordered_function[key] = function[key]
            ordered_function.update(
                {
                    key: value
                    for key, value in function.items()
                    if key not in ordered_function
                }
            )
            call["function"] = ordered_function
            ordered_call = {}
            for key in ("index", "type", "id", "function"):
                if key in call:
                    ordered_call[key] = call[key]
            ordered_call.update(
                {key: value for key, value in call.items() if key not in ordered_call}
            )
            prepared.append(ordered_call)
        return prepared

    @classmethod
    def _prepare_message(cls, message, marker=None, arguments_type="mapping"):
        message = deepcopy(message)
        if message.get("role") != "assistant":
            return message

        reasoning = message.get("reasoning")
        reasoning_content = message.get("reasoning_content")
        if reasoning and not reasoning_content:
            message["reasoning_content"] = reasoning
        elif reasoning_content and not reasoning:
            message["reasoning"] = reasoning_content
        elif reasoning is None and "reasoning_content" in message:
            message["reasoning"] = reasoning_content
        elif reasoning_content is None and "reasoning" in message:
            message["reasoning_content"] = reasoning

        if "tool_calls" in message and message["tool_calls"] is not None:
            message["tool_calls"] = cls._prepare_tool_calls(
                message["tool_calls"], marker=marker, arguments_type=arguments_type
            )
            if message["tool_calls"]:
                if message.get("content") == "":
                    message["content"] = None
                else:
                    message.setdefault("content", None)
            else:
                message.setdefault("content", "")
        elif "content" not in message:
            # The marker is already after reasoning, so this placeholder is truncated away.
            message["content"] = cls.placeholder if marker else ""

        ordered_message = {}
        for key in (
            "role",
            "reasoning",
            "reasoning_content",
            "content",
            "tool_calls",
            "finish_reason",
        ):
            if key in message:
                ordered_message[key] = message[key]
        ordered_message.update(
            {key: value for key, value in message.items() if key not in ordered_message}
        )
        return ordered_message

    def _apply_chat_template(self, messages, tools):
        kwargs = dict(tokenize=False, add_generation_prompt=False, tools=tools)
        rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        assert isinstance(rendered, str), type(rendered)
        return rendered

    @staticmethod
    def _has_tool_calls(messages):
        return any(message.get("tool_calls") for message in messages)

    def _render_with_native_arguments(self, messages, tools, marker=None):
        arguments_type = self._tool_arguments_type or "mapping"

        def prepare(arguments_type):
            return [
                self._prepare_message(
                    message,
                    marker=marker if message is messages[-1] else None,
                    arguments_type=arguments_type,
                )
                for message in messages
            ]

        try:
            return self._apply_chat_template(prepare(arguments_type), tools)
        except TypeError:
            if arguments_type == "string" or not self._has_tool_calls(messages):
                raise
            self._tool_arguments_type = "string"
            return self._apply_chat_template(prepare("string"), tools)

    def _render_messages(self, messages, tools):
        return self._render_with_native_arguments(messages, tools)

    def _render_partial_messages(self, messages, partial_message, tools):
        marker = f"{ON_PANDA_TRUNCATE_PREFIX}{uuid.uuid4().hex}"
        marked_message = self._append_partial_marker(partial_message, marker)
        rendered = self._render_with_native_arguments(
            messages + [marked_message], tools, marker=marker
        )
        marker_index = rendered.find(marker)
        assert marker_index != -1, rendered
        return rendered[:marker_index]

    def _encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _rendered_diff(self, base_text, corrected_text):
        return self._token_diff(self._encode(base_text), self._encode(corrected_text))

    @staticmethod
    def _token_diff(base_tokens, corrected_tokens):
        common_length = min(len(base_tokens), len(corrected_tokens))
        fork_token_index = next(
            (
                token_index
                for token_index in range(common_length)
                if base_tokens[token_index] != corrected_tokens[token_index]
            ),
            common_length,
        )
        return dict(
            fork_token_index=fork_token_index,
            replacement_tokens=corrected_tokens[fork_token_index:],
        )

    @staticmethod
    def _status(correction):
        status = correction.get("status")
        if "fork_message_index" not in correction:
            if status in ("parse_failed", "not_found", "no_op"):
                return status
            return "parse_failed"
        if correction["fork_message_index"] is None:
            return "is_good"
        if "corrected_messages" not in correction:
            return "parse_failed"
        assert isinstance(correction["fork_message_index"], int), correction
        return "applied"

    def _is_partial(self, correction, message, trajectory_role):
        if correction.get("status") == "partial":
            return True
        if message.get("finish_reason") in self.partial_finish_reasons:
            return True
        if trajectory_role == "pred":
            return message.get("finish_reason") not in self.complete_finish_reasons
        return False

    def _protocol_reward(self, correction, status):
        parse_and_locate = correction.get("parse_and_locate") or {}
        reward = correction.get("reward_with_feedback") or parse_and_locate.get(
            "reward_with_feedback"
        )
        if status == "not_found":
            return 0.5, reward or {}
        if reward and "format_reward" in reward:
            return float(reward["format_reward"]), reward
        if reward and {"parse_reward", "find_reward"} <= reward.keys():
            return (
                (float(reward["parse_reward"]) + float(reward["find_reward"])) / 2,
                reward,
            )
        if status in ("parse_failed", "no_op"):
            return 0.0, reward or {}
        return 1.0, reward or {}

    def _render_correction(self, base_messages, correction, tools, role):
        fork_message_index = correction["fork_message_index"]
        assert isinstance(fork_message_index, int), fork_message_index
        corrected_messages = correction["corrected_messages"]
        assert 0 <= fork_message_index < len(base_messages)
        assert 0 <= fork_message_index < len(corrected_messages)
        assert (
            corrected_messages[:fork_message_index]
            == base_messages[:fork_message_index]
        )
        render_context = [{"role": "user", "content": self.query_placeholder}]
        base_message = self._prepare_message(base_messages[fork_message_index])
        corrected_message = corrected_messages[fork_message_index]
        base_text = self._render_messages(render_context + [base_message], tools)
        if self._is_partial(correction, corrected_message, role):
            corrected_text = self._render_partial_messages(
                render_context, corrected_message, tools
            )
        else:
            corrected_text = self._render_messages(
                render_context + [self._prepare_message(corrected_message)],
                tools,
            )
        diff = self._rendered_diff(base_text, corrected_text)
        return diff

    def _replacement_reward(self, gt_diff, pred_diff, gt_is_partial=False):
        limit = self.max_replacement_tokens
        gt_tokens = gt_diff["replacement_tokens"]
        pred_tokens = pred_diff["replacement_tokens"]
        expected = gt_tokens[:limit] if limit is not None else gt_tokens
        if len(pred_tokens) < len(expected):
            return 0.0
        if len(gt_tokens) > len(expected):
            return float(pred_tokens[: len(expected)] == expected)
        if gt_is_partial:
            return float(pred_tokens[: len(expected)] == expected)
        return float(pred_tokens == expected)

    def _verify_pair(
        self,
        pred_trajectory,
        gt_trajectory,
        pred_correction,
        gt_correction,
        pred_correction_index,
        gt_correction_index,
    ):
        base_messages = pred_trajectory["messages"]
        tools = pred_trajectory.get("tools")
        pred_status = self._status(pred_correction)
        gt_status = self._status(gt_correction)
        pred_fork_message_index = (
            pred_correction["fork_message_index"]
            if pred_status not in ("is_good", "parse_failed", "not_found", "no_op")
            else None
        )
        gt_fork_message_index = (
            gt_correction["fork_message_index"]
            if gt_status not in ("is_good", "parse_failed", "not_found", "no_op")
            else None
        )
        pred_is_good = pred_status == "is_good"
        gt_is_good = gt_status == "is_good"
        format_reward, protocol_reward = self._protocol_reward(
            pred_correction, pred_status
        )
        if pred_status == "parse_failed":
            assert format_reward != 1.0, pred_correction
        is_good_cls_reward = float(pred_is_good == gt_is_good) if format_reward else 0.0
        comparable_statuses = ("is_good", "applied")
        fork_message_index_reward = float(
            pred_status in comparable_statuses
            and gt_status in comparable_statuses
            and pred_fork_message_index == gt_fork_message_index
        )

        location_reward = 0.0
        replacement_reward = 0.0
        location_feedback = "location mismatch"
        replacement_feedback = "replacement mismatch"
        if pred_status not in (
            "parse_failed",
            "not_found",
            "no_op",
        ) and gt_status not in (
            "parse_failed",
            "not_found",
            "no_op",
        ):
            if pred_is_good or gt_is_good:
                location_reward = float(pred_is_good and gt_is_good)
                replacement_reward = location_reward
                location_feedback = (
                    "location matched: is_good"
                    if location_reward
                    else "location mismatch: is_good"
                )
                replacement_feedback = (
                    "replacement skipped: is_good"
                    if replacement_reward
                    else "replacement mismatch: is_good"
                )
            elif fork_message_index_reward:
                gt_diff = self._render_correction(
                    base_messages, gt_correction, tools, "gt"
                )
                pred_diff = self._render_correction(
                    base_messages, pred_correction, tools, "pred"
                )
                location_reward = float(
                    gt_diff["fork_token_index"] == pred_diff["fork_token_index"]
                )
                replacement_reward = (
                    self._replacement_reward(
                        gt_diff,
                        pred_diff,
                        self._is_partial(
                            gt_correction,
                            gt_correction["corrected_messages"][gt_fork_message_index],
                            "gt",
                        ),
                    )
                    if location_reward
                    else 0.0
                )
                location_feedback = (
                    "location matched: token fork"
                    if location_reward
                    else "location mismatch: token fork"
                )
                replacement_feedback = (
                    "replacement matched: tokens"
                    if replacement_reward
                    else "replacement mismatch: tokens"
                )

        final_reward = (format_reward + location_reward + replacement_reward) / 3
        format_feedback = protocol_reward.get("format_feedback")
        if not format_feedback:
            format_feedback = (
                "format success: trajectory correction"
                if format_reward
                else "format failed: correction status"
            )
        feedback = "\n".join(
            [
                f"final_reward = {round(final_reward, 3)}",
                f"format_reward = {round(format_reward, 3)}",
                f"location_reward = {round(location_reward, 3)}",
                f"replacement_reward = {round(replacement_reward, 3)}",
                f"location_feedback: {location_feedback}",
                f"replacement_feedback: {replacement_feedback}",
            ]
        )
        reward_with_feedback = dict(
            final_reward=final_reward,
            format_reward=format_reward,
            location_reward=location_reward,
            replacement_reward=replacement_reward,
            is_good_cls_reward=is_good_cls_reward,
            fork_message_index_reward=fork_message_index_reward,
            pred_correction_index=pred_correction_index,
            gt_correction_index=gt_correction_index,
            pred_correction_indexes=[pred_correction_index],
            gt_correction_indexes=[gt_correction_index],
            multi_correction_gt_mode=self.multi_correction_gt_mode,
            tokenizer=dict(name_or_path=getattr(self.tokenizer, "name_or_path", "")),
            pred_status=pred_status,
            gt_status=gt_status,
            format_feedback=format_feedback,
            location_feedback=location_feedback,
            replacement_feedback=replacement_feedback,
            feedback=feedback,
        )
        for key in (
            "parse_reward",
            "parse_feedback",
            "find_reward",
            "find_feedback",
        ):
            if key in protocol_reward:
                reward_with_feedback[key] = protocol_reward[key]
        result = deepcopy(pred_correction.get("parse_and_locate") or {})
        for key in ("find_and_replace", "messages_location"):
            result[key] = deepcopy(pred_correction.get(key, result.get(key, {})))
        result["reward_with_feedback"] = reward_with_feedback
        return result

    def verify_one(
        self,
        pred_trajectory,
        gt_trajectory,
        *,
        pred_correction_index=0,
        gt_correction_index=0,
    ):
        assert pred_trajectory["messages"] == gt_trajectory["messages"]
        assert pred_trajectory.get("tools") == gt_trajectory.get("tools")
        pred_corrections = pred_trajectory["corrections"]
        gt_corrections = gt_trajectory["corrections"]
        assert pred_corrections and gt_corrections
        return self._verify_pair(
            pred_trajectory,
            gt_trajectory,
            pred_corrections[pred_correction_index],
            gt_corrections[gt_correction_index],
            pred_correction_index,
            gt_correction_index,
        )

    def verify(self, pred_trajectory, gt_trajectory):
        assert pred_trajectory["messages"] == gt_trajectory["messages"]
        assert pred_trajectory.get("tools") == gt_trajectory.get("tools")
        assert pred_trajectory["corrections"] and gt_trajectory["corrections"]
        results = [
            self.verify_one(
                pred_trajectory,
                gt_trajectory,
                pred_correction_index=pred_correction_index,
                gt_correction_index=gt_correction_index,
            )
            for pred_correction_index in range(len(pred_trajectory["corrections"]))
            for gt_correction_index in range(len(gt_trajectory["corrections"]))
        ]
        if self.multi_correction_gt_mode == "max":
            return [
                max(
                    results,
                    key=lambda result: result["reward_with_feedback"]["final_reward"],
                )
            ]
        return results


if __name__ == "__main__":
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
    codec = adapter
    gt_correction = adapter.build_correction_from_rejected_messages(rejected_msgs1)

    split = codec.special_tokens["split"]
    is_good = codec.special_tokens["is_good"]
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
            f"{split}{location_text}{split}{location_index + 1}{split}{replacement_token}{split}",
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
    ]

    for case_name, far_text, expected in far_text_cases:
        result = codec.compute_reward(
            rejected_msgs1, far_text, gt_correction=gt_correction
        )
        reward_res = result["reward_with_feedback"]
        assert all(
            [reward_res[k] == expected[k] for k in expected]
        ), f"{case_name}, {far_text}, {expected}\n\n{reward_res}"
        print(f"\n\n{case_name} passed: \n{far_text}\n{reward_res['feedback']}")

    gt_apply = adapter.apply(rejected_msgs1, far_text_gt)
    gt_correction = deepcopy(gt_apply["correction"])
    gt_correction.update(
        status="partial",
        fork_message_index=gt_correction["messages_location"]["path_keys"][0],
        corrected_messages=gt_apply["partial_messages"],
    )
    gt_trajectory = dict(
        messages=deepcopy(rejected_msgs1),
        tools=None,
        corrections=[gt_correction],
    )
    trajectory_verifier = onpanda.CorrectionVerifier(
        tokenizer=onpanda.utf8_tokenizer, max_replacement_tokens=2
    )
    for case_name, far_text, _ in far_text_cases:
        pred_apply = adapter.apply(rejected_msgs1, far_text)
        pred_correction = deepcopy(pred_apply["correction"])
        pred_correction["parse_and_locate"] = codec.parse_and_locate(
            rejected_msgs1, far_text
        )
        if pred_correction["find_and_replace"].get("is_good"):
            pred_correction.update(status="is_good", fork_message_index=None)
        elif (
            pred_correction["parse_and_locate"]["reward_with_feedback"]["parse_reward"]
            == 0.0
        ):
            pred_correction["status"] = "parse_failed"
        elif pred_correction["messages_location"].get("not_found"):
            pred_correction["status"] = "not_found"
        else:
            pred_correction.update(
                status="partial",
                fork_message_index=pred_correction["messages_location"]["path_keys"][0],
                corrected_messages=pred_apply["partial_messages"],
            )
        pred_trajectory = dict(
            messages=deepcopy(rejected_msgs1),
            tools=None,
            corrections=[pred_correction],
        )
        trajectory_result = trajectory_verifier.verify(pred_trajectory, gt_trajectory)[
            0
        ]
        assert {"find_and_replace", "messages_location"} <= trajectory_result.keys()
        if case_name == "case1_all_correct":
            assert trajectory_result["reward_with_feedback"]["final_reward"] == 1.0
        print(f"trajectory {case_name} passed:", trajectory_result)
