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
    from ..token_level_supervision_utils import unicode_tokenizer

correcting_sft_system_prompt_cn = """- 先前的 system prompt 只做评估用，不必再遵守
- 你本体是一个 GPT 架构的 LLM, 你现在的角色切换为了 token-level correcting model
- 目标是通过修改不恰当的 token 来优化已有的回答
- 你的任务是：
    1. 定位上述回答中，第一个不恰当的 token，即指出 “修改位置”
    2. 将“不恰当 token”修改为更加恰当的 token，使得基于 “恰当 token” 继续做补全能获得最好、最准确的答复
- Correcting 范围：多轮的情况下，只定位和修改上一轮（即最新轮）的答复中首个“不恰当 token”
- 由于你作为 LLM 只会输出文本，我们按照这个文本格式来输出你的 correcting 答复:
    - `<|split|>{location_tokens}<|split|>{location_index}<|split|>{replacement_token}<|split|>`
    - `<|split|>` 是分隔内容的 special token，且回答必须以 `<|split|>` 作为开头和结尾
    - `{location_tokens}`: 用来定位 “修改位置” 的一串 tokens
        - 其内容为从不恰当的 token 开始，持续生成，直到触发以下任意情况：
            1. 在所有模型输出的 tokens 中 (包括模型的历史输出) 被 `{location_tokens}` 匹配上的第一处位置正好就是 “修改位置” 
                - 此时的 `{location_index}` 应该为 0，并停止生成
                - 若第一匹配处不是 “修改位置”，则继续生成下一个 token 来做更加精准的定位
            2. `{location_tokens}` 长度达到 20 个 token，就该停止生成了
                - 但是，若最后的几个 token 不能被你自己 (correcting model) 的 tokenizer decode 为完整字符，需要突破 20 tokens 限制生成到能 decode 出完整字符为止
                - 若 20 个 token 都没法把 “修改位置” 准确定位，那就需要配合 `{location_index}` 来一起定位了
            3. 一轮结束了，即已经生成了 stop token: `<|stop|>`，也应该停止生成
    - `{location_index}` 表示在所有模型输出的 tokens 中, 能被 `{location_tokens}` 匹配上的所有位置中的第几个位置
        - 是一个 int 数值，从 0 开始计数，支持负数，和 Python list 的 index 一致
        - 当用负数表示 index 时的绝对值比正数 index 更加小的时候，`{location_index}` 就用负数表示
        - `{location_tokens}` 和 `{location_index}` 配合后，能在所有答复中共同定位一个唯一的位置，即 “第一个不恰当 token” 的位置
    - `{replacement_token}`: 更加恰当的 token，期望改为恰当 token 后，继续做补全能获得最好、最准确的答复。只需要一个 token 即可
    -  stop token: 上面的每一轮答复最后都有 stop token，需要的话，在 `{location_tokens}`,`{replacement_token}` 中使用 special token `<|stop|>` 来表示 stop token
        - 比如, 要续写最后一轮的答复 `<|split|><|stop|><|split|>-1<|split|>{continue token}<|split|>`
    - tokenizer 问题：
        - 你需要通过多输出 token 或提前输出 token 来避免潜在的 tokenizer decode 出不合规文本的问题。
        - 即多个 tokens 对应一个文本字符的情况下，要把多个 token 视为一个整体，使所有输出的 tokens 能和文本互相转换，而不要截断中间 token
    - 如果 Correcting 范围内的回答都没有问题，输出 `<|split|><|split|>`

## example 1:
USER:
列举 3 种水果：
ASSISTANT:
苹果、土豆、香蕉
期望的输出: “<|split|>土豆<|split|>0<|split|>西瓜<|split|>”

## example 2:
USER:
Just reply 2 times, Using "|" as a separator:
1;2;3;4;5;6;7;8;9;8;
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;
USER:
Reply again
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;

期望的输出: “<|split|>|1;2;3;4;5;6;7;8;9;8<|split|>-1<|split|><|stop|><|split|>”
- “第一个不恰当 token”处和其他 ASSISTANT 的回答有重复，所以会生成完整个 20 个 `{location_tokens}`
- `{location_index}` 用正数表示时为 2， 用负数为 -1，其中， -1 绝对值更加小，所以应该用 -1
- 此处 `{replacement_token}` 为 stop token"""

correcting_sft_system_prompt_default = correcting_sft_system_prompt_cn


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


class NextTokenPredictionAsCorrectingBuilder:
    def __init__(
        self,
        tokenizer=None,
        SPLIT_TOKEN="<|split|>",  # for qwen 2.5
        STOP_TOKEN="<|stop|>",
        max_location_tokens=20,
        scope_slice=(-1, None),  # TODO: slice of which messages can be correcting
    ):
        self.tokenizer = tokenizer or unicode_tokenizer
        self.SPLIT_TOKEN = SPLIT_TOKEN
        self.STOP_TOKEN = STOP_TOKEN
        self.max_location_tokens = max_location_tokens
        self.scope_slice = scope_slice

    def get_correcting_sft_system_prompt(self, language="cn"):
        if language == "cn":
            prompt = correcting_sft_system_prompt_cn
        else:
            prompt = correcting_sft_system_prompt_default
        return (
            prompt.replace("<|split|>", self.SPLIT_TOKEN)
            .replace("<|stop|>", self.STOP_TOKEN)
            .replace(" 20 ", f" {self.max_location_tokens} ")
        )

    def convert_token_level_to_unicode_location(self, rejected_msgs):
        """
        根据 rejected_msgs 中的 token_level 信息返回 unicode_location

        Args:
            rejected_msgs: 消息列表

        Returns:
            dict: {"message_index": int, "unicode_index": int}
        """
        # 查找首个有 token_level 的 assistant 消息
        for i, msg in enumerate(rejected_msgs):
            if msg["role"] == "assistant" and "token_level" in msg:
                token_level = msg["token_level"]
                unicode_location = token_level["rejected_text_unicode_range"][0]
                return {"message_index": i, "unicode_index": unicode_location}
        return {"not_found": True}

    def parser_ntp_as_correcting_text(self, ntp_as_correcting_text):
        mid_text = ntp_as_correcting_text.removeprefix(self.SPLIT_TOKEN).removesuffix(
            self.SPLIT_TOKEN
        )
        if mid_text:  # correcting
            splits = mid_text.split(self.SPLIT_TOKEN)
            # TODO: How to handle exception?
            assert len(splits) == 3, splits
            ntp_as_correcting = dict(
                zip(["location_text", "location_index", "replacement_token"], splits)
            )
            ntp_as_correcting["location_index"] = int(
                ntp_as_correcting["location_index"]
            )
        else:  # is_good
            ntp_as_correcting = dict(is_good=True, location_text="")
        return ntp_as_correcting

    def get_unicode_location(self, msgs, ntp_as_location=None):
        """
        根据 ntp_as_location 定位 unicode_location
        如果 ntp_as_location is None, 则从 msgs 必须是 correcting_sft, 会从最后一条消息中解析出 ntp_as_location
        """
        if ntp_as_location is None:
            sys_msg, correcting_msg = msgs[-2:]
            msgs = msgs[:-2]
            # ntp_as_correcting_gt = correcting_msg.get('correcting')
            ntp_as_correcting_text = mxlm.get_text_content(correcting_msg)
            ntp_as_location = self.parser_ntp_as_correcting_text(ntp_as_correcting_text)
            if ntp_as_location.get("is_good"):
                return dict(not_found=True, is_good=True)
        unicode_location = self._get_unicode_location(msgs, ntp_as_location)
        return unicode_location

    def _get_unicode_location(self, msgs, ntp_as_location):
        """
        Compute unicode_location by ntp_as_location in messages without token_level_info
        if Not found:
            return dict(not_found=True)

        用 for loop 遍历所有 assistant 消息，找到所有能匹配上 location_text 的位置
        如果能找到， 返回 location_index 对应的位置的 unicode_location
        否则返回 not_found=True
        """
        unicode_sequence_dic = self.messages_to_assistant_unicode_sequence(msgs)
        assistant_sequence = unicode_sequence_dic["assistant_sequence"]
        location_index = ntp_as_location["location_index"]
        location_text = ntp_as_location.get("location_text", "")
        assert location_text, ntp_as_location
        unicode_locations = []
        for message_index, assistant_content in zip(
            unicode_sequence_dic["assistant_indices"],
            assistant_sequence.split(self.STOP_TOKEN),
        ):

            assistant_content += self.STOP_TOKEN
            start = 0
            while True:
                index = assistant_content.find(location_text, start)
                if index == -1:
                    break
                unicode_location = dict(
                    message_index=message_index, unicode_index=index
                )
                unicode_locations.append(unicode_location)
                start = index + 1
        match_num = len(unicode_locations)

        if match_num and -match_num <= location_index and location_index < match_num:
            unicode_location = unicode_locations[location_index]
            unicode_location["match_num"] = match_num
            return unicode_location
        else:
            return dict(not_found=True, match_num=match_num)

    def messages_to_assistant_unicode_sequence(self, msgs, unicode_location=None):
        """
        Convert messages to a single text sequence, if unicode_location is given,
        also compute the sequence_index in the combined text sequence.

        Returns:
            update to unicode_location dict: {"assistant_sequence": str, "sequence_index": int (if unicode_location is given)}
        """

        # 收集所有assistant消息的内容，并记录其在原始消息中的索引
        assistant_contents = []
        assistant_indices = []
        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant":
                content = mxlm.get_text_content(msg["content"])
                # 添加隐藏的 STOP_TOKEN
                content += self.STOP_TOKEN
                # content += "\n\n-----\n\n" 会导致潜在的 tokenizer 粘连问题
                assistant_contents.append(content)
                assistant_indices.append(i)

        assistant_sequence = "".join(assistant_contents)
        if unicode_location is None:
            unicode_location = {}
        else:
            message_index = unicode_location["message_index"]
            target_unicode_index = unicode_location["unicode_index"]
            # 计算目标位置的unicode位置
            # 找到目标消息在assistant消息列表中的索引
            try:
                assistant_msg_idx = assistant_indices.index(message_index)
            except ValueError:
                raise ValueError(f"消息索引 {message_index} 不是 assistant 消息")

            current_index = 0
            for i in range(assistant_msg_idx):
                current_index += len(assistant_contents[i])
            sequence_index = current_index + target_unicode_index
            # unicode_location = deepcopy(unicode_location)
            unicode_location["sequence_index"] = sequence_index
        unicode_location["assistant_sequence"] = assistant_sequence
        unicode_location["assistant_indices"] = assistant_indices
        # print(unicode_location)
        return unicode_location

    def set_location_index(self, rejected_msgs, ntp_as_location, unicode_location):
        """
        在所有模型输出的 tokens 中查找 ntp_as_location.location_text 的所有匹配位置，
        返回对应的索引位置 ntp_as_location.location_index

        Args:
            rejected_msgs: 消息列表
            ntp_as_location: dict(location_text=...) or 要查找的字符串
            unicode_location: dict, 包含 message_index 和 unicode_index, 也可以包含 assistant_sequence 和 sequence_index

        Returns ntp_as_location:
            int: location_index，从0开始计数，负数表示从末尾倒数
        """
        if isinstance(ntp_as_location, str):
            ntp_as_location = dict(location_text=ntp_as_location)
        ntp_as_location = deepcopy(ntp_as_location)
        location_text = ntp_as_location["location_text"]
        if "assistant_sequence" not in unicode_location:
            unicode_location = self.messages_to_assistant_unicode_sequence(
                rejected_msgs, unicode_location
            )
        assistant_sequence = unicode_location["assistant_sequence"]
        sequence_index = unicode_location["sequence_index"]

        # 在所有assistant内容中查找location_text的所有匹配位置
        matches = []
        start = 0
        while True:
            index = assistant_sequence.find(location_text, start)
            if index == -1:
                break
            matches.append(index)
            start = index + 1

        location_index = None
        # 找到目标位置对应的匹配索引
        for idx, match_index in enumerate(matches):
            if match_index == sequence_index:
                # 如果负数的绝对值更小，使用负数表示
                negative_idx = idx - len(matches)
                if abs(negative_idx) < idx:
                    location_index = negative_idx
                else:
                    location_index = idx

        ntp_as_location.update(
            unicode_location=unicode_location, match_num=len(matches)
        )
        ntp_as_location["location_index"] = location_index
        if not len(matches):
            ntp_as_location["not_found"] = True
        return ntp_as_location

    def convert_rejected_content_to_ntp_as_location(self, rejected_msgs):
        """
        将 rejected_msgs 和 token_level_info 转换为 Next Token Prediction as location 格式

        - 获得 correcting 位置的 unicode_location
        - 从 unicode_location 处取 suffix 再 decode
        - 循环 next valid decodable 直到 location_index 为 0，或者 token 超长
        - 生成并返回 location_text 和 location_index

        Args:
            rejected_msgs: 消息列表

        Returns:
            dict: {"location_text": str, "location_index": int}
        """
        # 获取 unicode_location
        unicode_location = self.convert_token_level_to_unicode_location(rejected_msgs)
        message_index = unicode_location["message_index"]
        unicode_index = unicode_location["unicode_index"]

        content = mxlm.get_text_content(rejected_msgs[message_index]["content"])
        content_suffix = content[unicode_index:] + self.STOP_TOKEN
        suffix_tokens = self.tokenizer.encode(content_suffix, add_special_tokens=False)
        decodable_num = 0

        while True:
            decodable_res = next_decodable_num(
                suffix_tokens, decodable_num, self.tokenizer
            )
            decodable_num = decodable_res["next_num"]
            location_text = decodable_res["decoded_text"]
            ntp_as_location = self.set_location_index(
                rejected_msgs,
                location_text,
                unicode_location,
            )
            if ntp_as_location.get("not_found"):
                raise ValueError("无法定位到 location_text", ntp_as_location)
            location_index = ntp_as_location.get("location_index", None)
            if location_index == 0:
                break
            if decodable_num >= len(suffix_tokens):
                break
            if decodable_num >= self.max_location_tokens:
                break

        ntp_as_location["location_tokens"] = suffix_tokens[:decodable_num]

        if "asset_location_consistency":
            unicode_location2 = self.get_unicode_location(
                rejected_msgs, ntp_as_location
            )
            assert (
                unicode_location["message_index"] == unicode_location2["message_index"]
                and unicode_location["unicode_index"]
                == unicode_location2["unicode_index"]
            ), (
                "asset_location_consistency: "
                + str(unicode_location)
                + str(unicode_location2)
                + str(ntp_as_location)
            )
        return ntp_as_location

    def build_correcting_sft_by_token_level_SFT(
        self, msgs, is_good=None
    ):  # must be is_good SFT msgs or token_level_SFT msgs
        unicode_location = self.convert_token_level_to_unicode_location(msgs)

        sys_prompt_message = dict(
            role="system",
            content=self.get_correcting_sft_system_prompt(),
        )
        # double check
        if is_good is not None:
            assert bool(is_good) == bool(
                unicode_location.get("not_found")
            ), f"is_good must consistent with token_level_info, is_good: {is_good} != unicode_location: {unicode_location}"

        [msg.update(ignore_loss=True) for msg in msgs if msg["role"] == "assistant"]
        if unicode_location.get(
            "not_found"
        ):  # 没有 token_level 信息, 属于 is_good 的 SFT
            is_good_correcting_msg = dict(
                role="assistant",
                content=self.SPLIT_TOKEN * 2,
                correcting=dict(is_good=True, scope_slice=self.scope_slice),
            )
            correcting_sft = msgs + [sys_prompt_message, is_good_correcting_msg]

            return correcting_sft
        else:  # 有 token_level 信息, 属于 not is_good 的 token-level SFT
            token_level_msg = msgs[-1]
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
            rejected_msgs = msgs[:-1] + [rejected_msg]

            ntp_as_location = self.convert_rejected_content_to_ntp_as_location(
                rejected_msgs,
            )
            ntp_as_correcting = deepcopy(ntp_as_location)
            ntp_as_correcting.pop("unicode_location", None)
            replacement_text = (
                token_level_info["chosen_text"] or self.STOP_TOKEN
            )  # if chosen_text is empty mean chosen stop token
            ntp_as_correcting.update(
                replacement_text=replacement_text,
                is_good=False,
                scope_slice=self.scope_slice,
            )

            correcting_content = f"{self.SPLIT_TOKEN}{ntp_as_correcting['location_text']}{self.SPLIT_TOKEN}{ntp_as_correcting['location_index']}{self.SPLIT_TOKEN}{ntp_as_correcting['replacement_text']}{self.SPLIT_TOKEN}"
            correcting_msg = dict(
                role="assistant",
                content=correcting_content,
                correcting=ntp_as_correcting,
            )
            correcting_sft = rejected_msgs + [
                sys_prompt_message,
                correcting_msg,
            ]
        # import boxx.g
        return correcting_sft

    def apply_ntp_as_correcting(self, msgs, ntp_as_correcting: str | dict):
        if isinstance(ntp_as_correcting, str):
            ntp_as_correcting = self.parser_ntp_as_correcting_text(ntp_as_correcting)
        if ntp_as_correcting.get("is_good"):
            return dict(
                ntp_as_correcting=ntp_as_correcting,
            )
        unicode_location = self.get_unicode_location(msgs, ntp_as_correcting)
        if unicode_location.get("not_found"):
            return dict(
                ntp_as_correcting=ntp_as_correcting, unicode_location=unicode_location
            )
        else:
            msg_idx = unicode_location["message_index"]
            partial_msg = deepcopy(msgs[msg_idx])
            if isinstance(partial_msg["content"], list):
                assert all(
                    [d["type"] == "text" for d in partial_msg["content"]]
                ), partial_msg
                partial_msg["content"] = mxlm.get_text_content(partial_msg["content"])
            good_prefix = partial_msg["content"][: unicode_location["unicode_index"]]
            if self.STOP_TOKEN == ntp_as_correcting["replacement_token"]:
                # no need continue final message
                partial_msg["content"] = good_prefix
                partial_msg["finish_reason"] = "stop"
            else:
                partial_msg["content"] = (
                    good_prefix + ntp_as_correcting["replacement_token"]
                )
                if "finish_reason" in partial_msg:
                    del partial_msg["finish_reason"]
            partial_messages = msgs[:msg_idx] + [partial_msg]
            correction = dict(
                ntp_as_correcting=ntp_as_correcting,
                unicode_location=unicode_location,
                partial_messages=partial_messages,
            )
            return correction


if __name__ == "__main__":
    from boxx import *

    with mximport.inpkg():
        from ..test_utils import build_test_tokenizer, get_test_rejected_msgs1
        from ..parser import build_test_panda_tree

    panda_json_dir = "../../../on-panda-example-data/panda_json"
    tokenizer = build_test_tokenizer()
    # build_argkws = dict(tokenizer=unicode_tokenizer)
    build_argkws = dict(
        tokenizer=tokenizer,
        SPLIT_TOKEN="<|fim_pad|>",  # for qwen 2.5
        STOP_TOKEN="<|fim_suffix|>",
    )
    builder = NextTokenPredictionAsCorrectingBuilder(**build_argkws)

    # test next_decodable_num
    complex_emoji_text = "🧎🏿‍♂️‍➡️"
    decodable = next_decodable_num(tokenizer.encode(complex_emoji_text), 0, tokenizer)
    assert decodable["next_num"] != 1, decodable

    # test sample case
    rejected_msgs1, ntp_as_correcting_text_gt1 = get_test_rejected_msgs1()[:2]

    result1 = builder.convert_rejected_content_to_ntp_as_location(rejected_msgs1)
    assert result1["location_text"] == " potato", result1
    assert result1["location_index"] == 0, result1

    correction1 = builder.apply_ntp_as_correcting(
        rejected_msgs1, ntp_as_correcting_text_gt1
    )
    assert correction1["partial_messages"][-1]["content"] == "Apple, orange"
    assert (
        "finish_reason" not in correction1["partial_messages"][-1]
    ), "Should continue_final_message (no finish_reason)"

    # test correcting_sft extreme cases: chosen stop
    test_json = (
        f"{panda_json_dir}/2025-09-10_correcting_sft_tokenizer-Qwen2.5.panda.json"
    )
    panda_tree = build_test_panda_tree(test_json)
    correcting_sft2 = panda_tree.build_correcting_sft_data_v1(builder)[-1]
    correcting_content2 = correcting_sft2[-1]["content"]
    ntp_as_correcting_text_gt2 = "<|fim_pad|>|1;2;3;4;5;6;7;8;9;8<|fim_pad|>-1<|fim_pad|><|fim_suffix|><|fim_pad|>"
    assert correcting_content2 == ntp_as_correcting_text_gt2, correcting_content2
    correction2 = builder.apply_ntp_as_correcting(
        correcting_sft2[:-2], ntp_as_correcting_text_gt2
    )
    assert correction2["partial_messages"][-1]["finish_reason"] == "stop"

    # test correcting_sft extreme cases: chosen continue
    test_json3 = f"{panda_json_dir}/2025-09-11_correcting_sft_continue_tokenizer-Qwen2.5.panda.json"
    panda_tree3 = build_test_panda_tree(test_json3)
    correcting_sft3 = panda_tree3.build_correcting_sft_data_v1(builder)[-1]
    correcting_content3 = correcting_sft3[-1]["content"]
    assert (
        correcting_content3
        == "<|fim_pad|><|fim_suffix|><|fim_pad|>1<|fim_pad|>|<|fim_pad|>"
    ), correcting_content3

    # test single_char_repeat case: chosen stop
    test_json4 = (
        f"{panda_json_dir}/2025-09-12_single_char_repeat_tokenizer-Qwen2.5.panda.json"
    )
    panda_tree4 = build_test_panda_tree(test_json4)
    correcting_sft4 = panda_tree4.build_correcting_sft_data_v1(builder)[-1]
