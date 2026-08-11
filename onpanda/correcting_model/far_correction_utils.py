#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 21:19:59 2025

@author: DIYer22
"""

import mxlm
import mximport
from copy import deepcopy

with mximport.inpkg():
    from ..token_level_supervision_utils import build_tokenizer
    from .verifier import FindAndReplaceVerifier
    from ..response_templates import (
        build_messages_location,
        build_templated_char_index,
        flatten_messages_for_correcting,
    )
    from . import system_prompts


def next_decodable_num(tokens, current_num, tokenizer):
    """
    从 tokens 的 current_num 位置开始，找到下一个能被 tokenizer decode 出完整字符的 idx
    """
    for num in range(current_num + 1, len(tokens) + 1):  # number of tokens
        try:
            decoded_text = tokenizer.decode(tokens[0:num])
            if (
                tokenizer.encode(decoded_text, add_special_tokens=False)
                == tokens[0:num]
            ):
                return dict(next_num=num, decoded_text=decoded_text)
        except Exception:
            continue
    raise ValueError(
        "无法找到下一个可解码的位置",
        getattr(tokenizer, "name_or_path", "unknow_tokenizer"),
        tokens,
    )


class CorrectionAdapter:
    def __str__(self):
        if hasattr(self, "info"):
            from pprint import pformat

            return pformat(self.info)
        return object.__str__(self)

    __repr__ = __str__


class FindAndReplaceCorrectionAdapter(CorrectionAdapter):
    """
    Adapter of one model: its tokenizer, its response template and the FAR answer format.

    Used as the correcting model's adapter by default. When passed as `adapter_policy` of
    `CorrectingModel.correct_and_rollout`, only `tokenizer`, `response_template` and
    `max_replacement_tokens` matter, to render and token align the continuation prefix.
    """

    def __init__(
        self,
        tokenizer=None,
        special_tokens=None,
        max_location_tokens=20,
        max_replacement_tokens=1,
        tokenizer_aware=False,
        system_prompt_language=None,
        response_template=None,
    ):
        self.verifier = FindAndReplaceVerifier(
            special_tokens=special_tokens,
            response_template=response_template,
        )
        self.special_tokens = self.verifier.special_tokens
        self.response_template = self.verifier.response_template
        self.tokenizer = build_tokenizer(tokenizer)
        self.info = self.far_info = dict(
            tokenizer=dict(name_or_path=getattr(self.tokenizer, "name_or_path", "")),
            response_template=deepcopy(self.response_template.config),
            special_tokens=dict(self.special_tokens),
            max_location_tokens=max_location_tokens,
            max_replacement_tokens=max_replacement_tokens,
        )
        self.max_location_tokens = max_location_tokens
        self.max_replacement_tokens = max_replacement_tokens
        self.tokenizer_aware = tokenizer_aware
        self.system_prompt_language = system_prompt_language

    def build_correction_prompt(self, messages):
        """
        Select system prompt by adapter config.
        - system_prompt_language == "cn": use Chinese prompt by tokenizer_aware.
        - otherwise: use English prompt by tokenizer_aware.
        """
        if self.system_prompt_language == "cn":
            prompt = (
                system_prompts.far_tokenizer_aware_system_prompt_cn
                if self.tokenizer_aware
                else system_prompts.far_tokenizer_agnostic_system_prompt_cn
            )
        else:
            prompt = (
                system_prompts.far_tokenizer_aware_system_prompt_en
                if self.tokenizer_aware
                else system_prompts.far_tokenizer_agnostic_system_prompt_en
            )
        system_prompt = (
            prompt.replace("<|split|>", self.special_tokens["split"])
            .replace("<|stop|>", self.special_tokens["stop"])
            .replace("<|is_good|>", self.special_tokens["is_good"])
            .replace("<|reasoning|>", self.special_tokens["reasoning"])
            .replace(" 20 ", f" {self.max_location_tokens} ")
        )
        sys_prompt_message = dict(
            role="system",
            content=system_prompt,
        )
        return flatten_messages_for_correcting(messages, self.response_template) + [
            sys_prompt_message
        ]

    def truncate_replacement_token(self, replacement_token):
        replacement_tokens = self.tokenizer.encode(
            replacement_token, add_special_tokens=False
        )
        if len(replacement_tokens) <= self.max_replacement_tokens:
            return replacement_token
        if self.max_replacement_tokens <= 0:
            return ""
        decodable_res = next_decodable_num(
            replacement_tokens, self.max_replacement_tokens - 1, self.tokenizer
        )
        return decodable_res["decoded_text"]

    def convert_token_level_to_messages_location(self, rejected_messages):
        """
        根据 rejected_messages 中的 token_level 信息返回 messages_location
        """
        for message_index, message in enumerate(rejected_messages):
            if message["role"] == "assistant" and "token_level" in message:
                token_level = message["token_level"]
                replacement_token = None
                if "chosen_text" in token_level:
                    replacement_token = (
                        token_level["chosen_text"] or self.special_tokens["stop"]
                    )
                if "messages_location" in token_level:
                    messages_location = deepcopy(token_level["messages_location"])
                    if replacement_token is not None:
                        messages_location.update(
                            replacement_token=replacement_token,
                            is_good=False,
                        )
                    return messages_location
                char_index = token_level["rejected_text_unicode_range"][0]
                patch_length = token_level["rejected_text_unicode_range"][1]
                content = mxlm.get_text_content(message["content"])
                messages_location = dict(
                    path_keys=[message_index, "content"],
                    char_index=char_index,
                    patch_length=patch_length,
                    left5=content[max(0, char_index - 5) : char_index],
                    right5=content[char_index : char_index + 5],
                )
                if replacement_token is not None:
                    messages_location.update(
                        replacement_token=replacement_token,
                        is_good=False,
                    )
                return messages_location
        return dict(not_found=True)

    def set_location_index(
        self,
        rejected_messages,
        find_and_replace,
        target_message_index,
        target_char_index,
    ):
        """
        在所有模型输出的 templated prompt 中查找 find_and_replace.location_text 的所有匹配位置，
        返回对应的 find_and_replace.location_index
        """
        if isinstance(find_and_replace, str):
            find_and_replace = dict(location_text=find_and_replace)
        find_and_replace = deepcopy(find_and_replace)
        location_text = find_and_replace["location_text"]
        matches = []
        for templated_location in self.verifier._iter_assistant_templated_prompts(
            rejected_messages
        ):
            templated_prompt = templated_location["templated_prompt"]
            start = 0
            while True:
                index = templated_prompt.find(location_text, start)
                if index == -1:
                    break
                matches.append((templated_location["message_index"], index))
                start = index + 1

        location_index = None
        for idx, match in enumerate(matches):
            if match == (target_message_index, target_char_index):
                negative_idx = idx - len(matches)
                if abs(negative_idx) < idx:
                    location_index = negative_idx
                else:
                    location_index = idx
                break

        find_and_replace.update(
            match_num=len(matches),
            location_index=location_index,
        )
        if not matches or location_index is None:
            find_and_replace["not_found"] = True
        return find_and_replace

    def build_correction_from_rejected_messages(self, rejected_messages):
        """
        将 rejected_messages 的 token_level 信息转换为 correction
        location_text 沿本 adapter 自己 tokenizer 的可解码边界延伸，因为 correcting model
        必须精确复制自己的 token
        """
        messages_location = self.convert_token_level_to_messages_location(
            rejected_messages
        )
        templated_location = self.verifier.build_templated_location(
            rejected_messages, messages_location["path_keys"][0]
        )
        target_char_index = build_templated_char_index(
            templated_location, messages_location
        )
        # Structured positions inside a JSON escape or template scaffold project to a
        # canonical template boundary; keep the persisted context consistent with that snap.
        messages_location.update(
            build_messages_location(templated_location, target_char_index)
        )
        location_suffix = templated_location["templated_prompt"][target_char_index:]
        suffix_tokens = self.tokenizer.encode(location_suffix, add_special_tokens=False)
        decodable_num = 0

        while True:
            decodable_res = next_decodable_num(
                suffix_tokens, decodable_num, self.tokenizer
            )
            decodable_num = decodable_res["next_num"]
            location_text = decodable_res["decoded_text"]
            find_and_replace = self.set_location_index(
                rejected_messages,
                location_text,
                templated_location["message_index"],
                target_char_index,
            )
            if find_and_replace.get("not_found"):
                raise ValueError("无法定位到 location_text", find_and_replace)
            location_index = find_and_replace.get("location_index", None)
            if location_index == 0:
                break
            if decodable_num >= len(suffix_tokens):
                break
            if decodable_num >= self.max_location_tokens:
                break

        find_and_replace["location_tokens"] = suffix_tokens[:decodable_num]
        if "assert_location_consistency":
            messages_location2 = self.verifier.locate(
                rejected_messages, find_and_replace
            )
            assert (
                messages_location["path_keys"] == messages_location2["path_keys"]
                and messages_location["char_index"] == messages_location2["char_index"]
            ), (
                "assert_location_consistency: "
                + str(messages_location)
                + str(messages_location2)
                + str(find_and_replace)
            )
        replacement_token = messages_location.get("replacement_token")
        if replacement_token is not None:
            find_and_replace.update(
                replacement_token=replacement_token,
                is_good=False,
            )
        find_and_replace["far_text"] = (
            f"{self.special_tokens['split']}{find_and_replace['location_text']}"
            f"{self.special_tokens['split']}"
            f"{find_and_replace['location_index']}{self.special_tokens['split']}"
            f"{find_and_replace['replacement_token']}{self.special_tokens['split']}"
        )
        return dict(
            messages_location=messages_location,
            find_and_replace=find_and_replace,
        )

    def build_correction_data_from_token_level(
        self, messages, is_good=None
    ):  # must be is_good SFT msgs or token_level_SFT msgs
        messages = deepcopy(messages)
        messages_location = self.convert_token_level_to_messages_location(messages)
        if is_good is not None:
            assert bool(is_good) == bool(
                messages_location.get("not_found")
            ), f"is_good must consistent with token_level_info, is_good: {is_good} != messages_location: {messages_location}"

        [msg.update(ignore_loss=True) for msg in messages if msg["role"] == "assistant"]
        if messages_location.get("not_found"):
            is_good_messages_location = dict(not_found=True, is_good=True)
            is_good_find_and_replace = dict(
                is_good=True,
                location_text="",
                location_index=0,
                replacement_token="",
                far_text=(
                    f"{self.special_tokens['split']}{self.special_tokens['is_good']}"
                    f"{self.special_tokens['split']}"
                ),
            )
            is_good_correction_msg = dict(
                role="assistant",
                content=is_good_find_and_replace["far_text"],
                correction=dict(
                    messages_location=is_good_messages_location,
                    find_and_replace=is_good_find_and_replace,
                    far_info=deepcopy(self.far_info),
                ),
            )
            return self.build_correction_prompt(messages) + [is_good_correction_msg]

        token_level_msg = messages[-1]
        token_level_info = token_level_msg["token_level"]
        rejected_content_chunks = token_level_info.pop("rejected_content")
        rejected_channel_text = token_level_info.pop("rejected_channel_text", None)
        token_level_info["chosen_content"] = token_level_msg["content"]

        rejected_content_str = mxlm.get_text_content(rejected_content_chunks)
        # The chunks are the flattened response text, so parse them back into channels, otherwise
        # this template would flatten an already flattened message a second time.
        rejected_msg = dict(
            self.response_template.parse(
                rejected_content_str, tools=messages[0].get("tools")
            ),
            ignore_loss=True,
            token_level=token_level_info,
        )
        if rejected_channel_text is not None:
            channel = rejected_msg
            for key in token_level_info["rejected_messages_location"]["path_keys"][
                1:-1
            ]:
                channel = channel[key]
            channel[token_level_info["rejected_messages_location"]["path_keys"][-1]] = (
                rejected_channel_text
            )
        rejected_msg.setdefault(
            "finish_reason", token_level_info.get("rejected_finish_reason", "")
        )
        if "rejected_messages_location" in token_level_info:
            token_level_info["messages_location"] = token_level_info.pop(
                "rejected_messages_location"
            )
        rejected_messages = messages[:-1] + [rejected_msg]

        correction = self.build_correction_from_rejected_messages(
            rejected_messages,
        )
        find_and_replace = correction["find_and_replace"]
        messages_location = correction["messages_location"]
        correction_msg = dict(
            role="assistant",
            content=find_and_replace["far_text"],
            correction=dict(
                messages_location=messages_location,
                find_and_replace=find_and_replace,
                far_info=deepcopy(self.far_info),
            ),
        )
        far_correction = self.build_correction_prompt(rejected_messages) + [
            correction_msg
        ]
        return far_correction

    def apply(self, messages, correction_or_far_text, tools=None, adapter_policy=None):
        """
        Cut in the response template's text space and parse back, so a replacement can end any
        channel and every downstream channel is truncated by construction. adapter_policy decides
        whether the correction survives the policy's own template, defaults to this adapter.
        """
        if isinstance(correction_or_far_text, str):
            parse_result = self.verifier.parse(correction_or_far_text)
            find_and_replace = parse_result["find_and_replace"]
            correction = dict(
                find_and_replace=find_and_replace,
                reward_with_feedback=parse_result["reward_with_feedback"],
            )
        else:
            correction = deepcopy(correction_or_far_text)
            if "find_and_replace" in correction:
                find_and_replace = correction["find_and_replace"]
            elif "replacement_token" in correction:
                find_and_replace = correction
                correction = dict(find_and_replace=find_and_replace)
            else:
                assert (
                    "messages_location" not in correction
                ), "apply(messages, messages_location) is not supported. Please provide find_and_replace with replacement_token."
                raise AssertionError(
                    f"correction_or_far_text is not a valid correction opreation: \n{correction_or_far_text}"
                )

        if find_and_replace.get("is_good"):
            correction.setdefault(
                "messages_location", self.verifier.locate(messages, find_and_replace)
            )
            return dict(
                correction=correction,
                partial_messages=messages,
            )

        assert (
            "replacement_token" in find_and_replace
        ), f"`replacement_token` not found in find_and_replace: {find_and_replace}"

        if "messages_location" in correction:
            # A ground truth correction only carries its structured location.
            messages_location = correction["messages_location"]
            if messages_location.get("not_found"):
                return dict(
                    correction=correction,
                    partial_messages=messages,
                )
            templated_location = self.verifier.build_templated_location(
                messages, messages_location["path_keys"][0]
            )
            templated_char_index = build_templated_char_index(
                templated_location, messages_location
            )
        else:
            templated_location = self.verifier.locate_templated(
                messages, find_and_replace
            )
            if templated_location.get("not_found"):
                correction["messages_location"] = templated_location
                return dict(
                    correction=correction,
                    partial_messages=messages,
                )
            messages_location = self.verifier.build_messages_location(
                templated_location, find_and_replace
            )
            correction["messages_location"] = messages_location
            templated_char_index = templated_location["templated_char_index"]

        normalized_replacement_token, has_stop_token = (
            self.verifier._normalize_replacement_token(
                find_and_replace["replacement_token"]
            )
        )
        # No truncation here: the replacement lives in this template's marker space, where a cut
        # inside a marker means the intermediate representation stops parsing. Fitting it to a
        # token grid is the policy's job, in build_partial_templated_prompt.
        replacement = normalized_replacement_token
        if has_stop_token:
            replacement += self.special_tokens["stop"]
        templated_prompt = templated_location["templated_prompt"]
        message_index = templated_location["message_index"]
        partial_message = self.response_template.parse(
            templated_prompt[:templated_char_index] + replacement,
            messages=messages,
            tools=tools,
        )
        policy_templated = (adapter_policy or self).build_partial_templated_prompt(
            messages[message_index], partial_message
        )
        if not policy_templated["replacement"] and (
            partial_message.get("finish_reason") not in ("stop", "tool_calls")
            or (
                messages[message_index].get("finish_reason") in ("stop", "tool_calls")
                and policy_templated["templated_prompt"]
                == (adapter_policy or self).response_template.apply(
                    messages[message_index]
                )["templated_prompt"]
            )
        ):
            # A no-op correction: continuing it would reproduce the rejected response. Either the
            # replacement only repeats the rejected text, or the policy's response template cannot
            # express it, e.g. an opened but still empty tool call channel. Let the caller retry.
            messages_location.update(
                not_found=True,
                find_feedback="no-op correction: partial response is a prefix of the rejected one",
            )
            return dict(
                correction=correction,
                partial_messages=messages,
            )

        partial_messages = deepcopy(messages[:message_index]) + [partial_message]
        return dict(
            correction=correction,
            partial_messages=partial_messages,
        )

    def build_partial_templated_prompt(self, rejected_message, partial_message):
        """
        Render the corrected partial message for continuation, then align the correction to this
        model's own token boundary: diff the complete token sequences and keep only the first
        `max_replacement_tokens` tokens after their fork.

        An empty replacement means this template's round trip normalized the correction away.
        """
        templated_prompt = self.response_template.apply(partial_message)[
            "templated_prompt"
        ]
        rejected_templated_prompt = self.response_template.apply(rejected_message)[
            "templated_prompt"
        ]
        templated_tokens = self.tokenizer.encode(
            templated_prompt, add_special_tokens=False
        )
        rejected_templated_tokens = self.tokenizer.encode(
            rejected_templated_prompt, add_special_tokens=False
        )
        common_length = min(len(templated_tokens), len(rejected_templated_tokens))
        fork_token_index = next(
            (
                token_index
                for token_index in range(common_length)
                if templated_tokens[token_index]
                != rejected_templated_tokens[token_index]
            ),
            common_length,
        )
        if (
            fork_token_index == len(templated_tokens)
            or self.max_replacement_tokens <= 0
        ):
            return dict(
                templated_prompt=self.tokenizer.decode(
                    templated_tokens[:fork_token_index]
                ),
                replacement="",
            )
        target_token_num = min(
            fork_token_index + self.max_replacement_tokens, len(templated_tokens)
        )
        decodable_res = next_decodable_num(
            templated_tokens, target_token_num - 1, self.tokenizer
        )
        replacement = self.tokenizer.decode(
            templated_tokens[fork_token_index : decodable_res["next_num"]]
        )
        return dict(
            templated_prompt=decodable_res["decoded_text"],
            replacement=replacement,
        )


class NextTokenPredictionAsCorrectingBuilder:
    def __init__(self, *args, **kwargs):
        assert False, (
            "NextTokenPredictionAsCorrectingBuilder has been removed. "
            "Please downgrade to onpanda<=0.0.10, or switch to "
            "FindAndReplaceCorrectionAdapter."
        )


def test_reasoning_and_tool_calls_correcting():
    """Correct every channel of a reasoning tool calling response, without any API call."""
    adapter = FindAndReplaceCorrectionAdapter(max_replacement_tokens=20)
    split = adapter.special_tokens["split"]
    stop = adapter.special_tokens["stop"]
    reasoning_marker = adapter.special_tokens["reasoning"]

    def build_far_text(location_text, replacement_token, location_index=0):
        return f"{split}{location_text}{split}{location_index}{split}{replacement_token}{split}"

    reasoning_messages = [
        dict(role="user", content="1+1=?"),
        dict(
            role="assistant",
            reasoning="1+1 gets 3",
            content="The answer is 3.",
            finish_reason="stop",
        ),
    ]
    # Cut inside reasoning: content and finish_reason are truncated by construction.
    partial_message = adapter.apply(reasoning_messages, build_far_text("3", "2"))[
        "partial_messages"
    ][-1]
    assert partial_message == dict(
        role="assistant", reasoning="1+1 gets 2"
    ), partial_message
    # The replacement can end thinking, which no structured replacement could express.
    reasoning_end_message = adapter.apply(
        reasoning_messages, build_far_text(" gets 3", reasoning_marker)
    )["partial_messages"][-1]
    assert reasoning_end_message == dict(
        role="assistant", reasoning="1+1", content="", finish_reason="reasoning_end"
    ), reasoning_end_message

    tool_call_messages = [
        dict(role="user", content="read /tmp/a.txt"),
        dict(
            role="assistant",
            content="",
            tool_calls=[
                dict(
                    index=0,
                    type="function",
                    id="functions.read_file:0",
                    function=dict(name="read_file", arguments='{"path": "/tmp/b.txt"}'),
                )
            ],
            finish_reason="tool_calls",
        ),
    ]
    # Cut inside tool call arguments, and close the whole response with the stop token.
    correction = adapter.apply(
        tool_call_messages, build_far_text('b.txt"}', f'a.txt"}}{stop}')
    )
    assert correction["correction"]["messages_location"]["path_keys"] == [
        1,
        "tool_calls",
        0,
        "function",
        "arguments",
    ], correction["correction"]["messages_location"]
    corrected_tool_call = correction["partial_messages"][-1]["tool_calls"][0]
    assert (
        corrected_tool_call["function"]["arguments"] == '{"path": "/tmp/a.txt"}'
    ), corrected_tool_call
    assert corrected_tool_call["id"] == "functions.read_file:0", corrected_tool_call
    assert (
        correction["partial_messages"][-1]["finish_reason"] == "tool_calls"
    ), correction

    # A no-op correction is retryable instead of silently rolling out the rejected response.
    no_op_messages = [
        dict(role="user", content="Name three kinds of fruit:"),
        dict(role="assistant", content="Apple, potato, banana.", finish_reason="stop"),
    ]
    no_op_location = adapter.apply(no_op_messages, build_far_text(" potato", " p"))[
        "correction"
    ]["messages_location"]
    assert (
        no_op_location.get("not_found") and "no-op" in no_op_location["find_feedback"]
    ), no_op_location

    # The policy renders the same partial message with its own template, and the fork against
    # the rejected response is the text the policy has to continue from.
    adapter_policy = FindAndReplaceCorrectionAdapter(
        response_template=dict(name_or_path="Qwen/Qwen3.6-35B-A3B")
    )
    partial_tool_call_message = dict(
        role="assistant",
        content="",
        tool_calls=[
            dict(index=0, function=dict(name="read_file", arguments='{"path": "/tmp/a'))
        ],
    )
    partial_templated = adapter_policy.build_partial_templated_prompt(
        tool_call_messages[-1], partial_tool_call_message
    )
    assert partial_templated["templated_prompt"] == (
        "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/a"
    ), partial_templated
    assert partial_templated["replacement"] == "a", partial_templated
    # The correction is aligned to the policy's own token boundary, which may cut template
    # scaffolding short: a prefix ending at `\n` still guides the policy to close thinking.
    reasoning_partial_templated = adapter_policy.build_partial_templated_prompt(
        dict(
            role="assistant", reasoning="think", content="answer", finish_reason="stop"
        ),
        dict(
            role="assistant", reasoning="thi", content="", finish_reason="reasoning_end"
        ),
    )
    assert reasoning_partial_templated == dict(
        templated_prompt="<think>\nthi\n", replacement="\n"
    ), reasoning_partial_templated

    # The correcting prompt flattens every channel, and tool responses merge into one user turn.
    tool_response_messages = tool_call_messages + [
        dict(role="tool", tool_call_id="functions.read_file:0", content="hello"),
        dict(role="tool", tool_call_id="functions.read_file:1", content="world"),
    ]
    correction_prompt = adapter.build_correction_prompt(tool_response_messages)
    assert [message["role"] for message in correction_prompt] == [
        "user",
        "assistant",
        "user",
        "system",
    ], correction_prompt
    assert correction_prompt[1]["content"].endswith(
        '<|ON_PANDA_CALL_ARGUMENTS|>{"path": "/tmp/b.txt"}' + stop
    ), correction_prompt[1]
    assert (
        correction_prompt[2]["content"].count("<|ON_PANDA_TOOL_RESPONSE|>") == 2
    ), correction_prompt[2]
    return 7


def test_agent_panda_json_far_correction(far_adapter, panda_json):
    """
    Every ground truth FAR of an agent trajectory must score a perfect reward when fed back,
    which closes the loop parser -> ground truth FAR -> locate -> apply -> reward across the
    reasoning, tool call and content channels.
    """
    with mximport.inpkg():
        from ..parser import build_test_panda_tree
        from ..utils import remove_msgs_after_last_response_role

    template = far_adapter.response_template
    far_corrections = build_test_panda_tree(panda_json).build_far_correction_data_v1(
        far_adapter
    )
    located_channels = set()
    for far_correction in far_corrections:
        gt_correction = far_correction[-1]["correction"]
        # The sample stores flattened assistant messages, so parse the channels back to locate.
        messages = [
            (
                dict(message, **template.parse(message["content"]))
                if message["role"] == "assistant"
                else message
            )
            for message in remove_msgs_after_last_response_role(far_correction[:-2])
        ]
        far_text = gt_correction["find_and_replace"]["far_text"]
        reward = far_adapter.verifier.compute_reward(messages, far_text, gt_correction)
        assert reward["reward_with_feedback"]["final_reward"] == 1.0, (
            far_text,
            reward["reward_with_feedback"]["feedback"],
        )
        path_keys = gt_correction["messages_location"].get("path_keys")
        if path_keys:
            located_channels.add(tuple(path_keys[1:]))
    assert located_channels == {
        ("reasoning",),
        ("content",),
        ("tool_calls", 0, "function", "arguments"),
    }, located_channels
    return len(far_corrections)


if __name__ == "__main__":
    from boxx import *

    with mximport.inpkg():
        from ..test_utils import build_test_tokenizer, get_test_rejected_msgs1
        from ..parser import build_test_panda_tree

    panda_json_dir = "../../../on-panda-example-data/panda_json"
    tokenizer = build_test_tokenizer()
    print(
        "test_reasoning_and_tool_calls_correcting passed:",
        test_reasoning_and_tool_calls_correcting(),
    )
    build_argkws = dict(
        tokenizer=tokenizer,
        special_tokens=dict(
            split="<|fim_pad|>",  # for qwen 2.5
            stop="<|fim_suffix|>",
            is_good="<|fim_prefix|>",
            reasoning="<|fim_middle|>",
        ),
    )
    far_adapter = FindAndReplaceCorrectionAdapter(**build_argkws)

    # test next_decodable_num
    complex_emoji_text = "🧎🏿‍♂️‍➡️"
    decodable = next_decodable_num(tokenizer.encode(complex_emoji_text), 0, tokenizer)
    assert decodable["next_num"] != 1, decodable

    # test sample case
    rejected_msgs1, far_text_gt1 = get_test_rejected_msgs1()[:2]

    result1 = far_adapter.build_correction_from_rejected_messages(rejected_msgs1)
    assert result1["find_and_replace"]["location_text"] == " potato", result1
    assert result1["find_and_replace"]["location_index"] == 0, result1

    correction1 = far_adapter.apply(rejected_msgs1, far_text_gt1)
    assert correction1["partial_messages"][-1]["content"] == "Apple, orange"
    assert (
        "finish_reason" not in correction1["partial_messages"][-1]
    ), "Should continue_final_message (no finish_reason)"

    # test far_correction extreme cases: chosen stop
    test_json = (
        f"{panda_json_dir}/2025-09-10_correcting_sft_tokenizer-Qwen2.5.panda.json"
    )
    panda_tree = build_test_panda_tree(test_json)
    far_correction2 = panda_tree.build_far_correction_data_v1(far_adapter)[-1]
    correcting_content2 = far_correction2[-1]["content"]
    far_text_gt2 = "<|fim_pad|>|1;2;3;4;5;6;7;8;9;8<|fim_pad|>-1<|fim_pad|><|fim_suffix|><|fim_pad|>"
    assert correcting_content2 == far_text_gt2, correcting_content2
    correction2 = far_adapter.apply(far_correction2[:-2], far_text_gt2)
    assert correction2["partial_messages"][-1]["finish_reason"] == "stop"

    # test far_correction extreme cases: chosen continue
    test_json3 = f"{panda_json_dir}/2025-09-11_correcting_sft_continue_tokenizer-Qwen2.5.panda.json"
    panda_tree3 = build_test_panda_tree(test_json3)
    far_correction3 = panda_tree3.build_far_correction_data_v1(far_adapter)[-1]
    correcting_content3 = far_correction3[-1]["content"]
    assert (
        correcting_content3
        == "<|fim_pad|><|fim_suffix|><|fim_pad|>1<|fim_pad|>|<|fim_pad|>"
    ), correcting_content3

    # test single_char_repeat case: chosen stop
    test_json4 = (
        f"{panda_json_dir}/2025-09-12_single_char_repeat_tokenizer-Qwen2.5.panda.json"
    )
    panda_tree4 = build_test_panda_tree(test_json4)
    far_correction4 = panda_tree4.build_far_correction_data_v1(far_adapter)[-1]

    # test agent trajectory: reasoning, tool call arguments and content channels
    print(
        "test_agent_panda_json_far_correction passed:",
        test_agent_panda_json_far_correction(
            far_adapter,
            f"{panda_json_dir}/2026-06-25_agent_example_template-K2.panda.json",
        ),
    )
