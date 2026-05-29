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
    from ..utils import get_content_by_path_keys
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
    def __init__(
        self,
        tokenizer=None,
        special_tokens=None,
        max_location_tokens=20,
        max_replacement_tokens=1,
        tokenizer_aware=False,
        system_prompt_language=None,
    ):
        self.verifier = FindAndReplaceVerifier(
            special_tokens=special_tokens,
        )
        self.special_tokens = self.verifier.special_tokens
        self.tokenizer = build_tokenizer(tokenizer)
        self.info = self.far_info = dict(
            tokenizer=dict(name_or_path=getattr(self.tokenizer, "name_or_path", "")),
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
        return messages + [sys_prompt_message]

    def _set_by_path(self, data, path_keys, value):
        target = data
        for key in path_keys[:-1]:
            target = target[key]
        target[path_keys[-1]] = value

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
        self, rejected_messages, find_and_replace, messages_location
    ):
        """
        在所有模型输出文本中查找 find_and_replace.location_text 的所有匹配位置，
        返回对应的 find_and_replace.location_index
        """
        if isinstance(find_and_replace, str):
            find_and_replace = dict(location_text=find_and_replace)
        find_and_replace = deepcopy(find_and_replace)
        location_text = find_and_replace["location_text"]
        matches = []
        for path_keys, text in self.verifier._iter_assistant_text_locations(
            rejected_messages
        ):
            search_scope = text + self.special_tokens["stop"]
            start = 0
            while True:
                index = search_scope.find(location_text, start)
                if index == -1:
                    break
                matches.append((path_keys, index))
                start = index + 1

        target_path_keys = messages_location["path_keys"]
        target_char_index = messages_location["char_index"]
        location_index = None
        for idx, (path_keys, char_index) in enumerate(matches):
            if path_keys == target_path_keys and char_index == target_char_index:
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
        """
        messages_location = self.convert_token_level_to_messages_location(
            rejected_messages
        )
        path_keys = messages_location["path_keys"]
        char_index = messages_location["char_index"]
        content = get_content_by_path_keys(rejected_messages, path_keys)
        if isinstance(content, list):
            assert all([d["type"] == "text" for d in content]), rejected_messages
            content = mxlm.get_text_content(content)
        content_suffix = content[char_index:] + self.special_tokens["stop"]
        suffix_tokens = self.tokenizer.encode(content_suffix, add_special_tokens=False)
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
                messages_location,
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
        token_level_info["chosen_content"] = token_level_msg["content"]

        rejected_content_str = mxlm.get_text_content(rejected_content_chunks)
        rejected_msg = dict(
            role="assistant",
            ignore_loss=True,
            content=rejected_content_str,
            finish_reason=token_level_info.get("rejected_finish_reason", ""),
            token_level=token_level_info,
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

    def apply(self, messages, correction_or_far_text):
        if isinstance(correction_or_far_text, str):
            correction = self.verifier.parse_and_locate(
                messages, correction_or_far_text
            )
            find_and_replace = correction["find_and_replace"]
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

        if "messages_location" not in correction:
            correction["messages_location"] = self.verifier.locate(
                messages, find_and_replace
            )

        if find_and_replace.get("is_good"):
            return dict(
                correction=correction,
                partial_messages=messages,
            )

        assert (
            "replacement_token" in find_and_replace
        ), f"`replacement_token` not found in find_and_replace: {find_and_replace}"
        messages_location = correction["messages_location"]
        if messages_location.get("not_found"):
            return dict(
                correction=correction,
                partial_messages=messages,
            )

        path_keys = messages_location["path_keys"]
        char_index = messages_location["char_index"]
        replacement_token = find_and_replace["replacement_token"]
        normalized_replacement_token, has_stop_token = (
            self.verifier._normalize_replacement_token(replacement_token)
        )
        partial_messages = deepcopy(messages[: path_keys[0] + 1])

        field_text = get_content_by_path_keys(partial_messages, path_keys)
        if isinstance(field_text, list):
            assert all([d["type"] == "text" for d in field_text]), partial_messages
            field_text = mxlm.get_text_content(field_text)

        good_prefix = field_text[:char_index]
        corrected_field_text = good_prefix + self.truncate_replacement_token(
            normalized_replacement_token
        )
        self._set_by_path(partial_messages, path_keys, corrected_field_text)

        if path_keys[-1] == "content":
            if has_stop_token:
                partial_messages[-1]["finish_reason"] = "stop"
            elif "finish_reason" in partial_messages[-1]:
                del partial_messages[-1]["finish_reason"]

        return dict(
            correction=correction,
            partial_messages=partial_messages,
        )


class NextTokenPredictionAsCorrectingBuilder:
    def __init__(self, *args, **kwargs):
        assert False, (
            "NextTokenPredictionAsCorrectingBuilder has been removed. "
            "Please downgrade to onpanda<=0.0.10, or switch to "
            "FindAndReplaceCorrectionAdapter."
        )


if __name__ == "__main__":
    from boxx import *

    with mximport.inpkg():
        from ..test_utils import build_test_tokenizer, get_test_rejected_msgs1
        from ..parser import build_test_panda_tree

    panda_json_dir = "../../../on-panda-example-data/panda_json"
    tokenizer = build_test_tokenizer()
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
