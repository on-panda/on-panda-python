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
    from ..response_templates.partial_json import parse_partial_json_object
    from ..token_level_supervision_utils import build_tokenizer

ON_PANDA_PLACEHOLDER = "ON_PANDA_PLACEHOLDER"
ON_PANDA_TRUNCATE_PREFIX = "ON_PANDA_TRUNCATE_"
ON_PANDA_INVALID_ARGUMENTS = '{"ON_PANDA_PLACEHOLDER":"ON_PANDA_PLACEHOLDER"}'


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
    from onpanda.correcting_model.far_correction_utils import (
        FindAndReplaceCorrectionAdapter,
    )
    from onpanda.test_utils import build_test_tokenizer, get_test_far_text_cases

    tokenizer = build_test_tokenizer()
    adapter = FindAndReplaceCorrectionAdapter(
        tokenizer=tokenizer,
        special_tokens=dict(
            split="<|fim_pad|>",
            stop="<|fim_suffix|>",
            is_good="<|fim_prefix|>",
            reasoning="<|fim_middle|>",
        ),
    )
    far_text_cases = get_test_far_text_cases(adapter)
    verifier = CorrectionVerifier(tokenizer=tokenizer, max_replacement_tokens=2)
    for case in far_text_cases:
        verify_results = verifier.verify(case["pred_trajectory"], case["gt_trajectory"])
        assert len(verify_results) == 1
        verification_result = verify_results[0]
        assert {"find_and_replace", "messages_location"} <= verification_result.keys()
        reward_res = verification_result["reward_with_feedback"]
        assert all(
            reward_res[key] == expected
            for key, expected in case["expected_rewards"].items()
        ), f"{case['name']}, {case['pred_far']}, {case['expected_rewards']}\n\n{reward_res}"
        print(f"\n\n{case['name']} passed: {reward_res['feedback']}")
