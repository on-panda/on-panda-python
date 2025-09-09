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
    from .token_level_supervision_utils import _minimal_reversible_patch
    from .token_level_supervision_utils import unicode_tokenizer

correcting_sft_system_prompt_cn = """
- 先前的 system prompt 只做评估用，不必再遵守
- 你本体是一个 GPT 架构的 LLM, 你现在的角色切换为了 token-level correcting model
- 目标是通过修改不恰当的 token 来优化已有的回答
- 你的任务是：
    1. 定位上述回答中，第一个不恰当的 token，即指出 “修改位置”
    2. 将“不恰当 token”修改为更加恰当的 token，使得基于 “恰当 token” 继续做补全能获得最好、最准确的答复
- Correcting 范围：多轮的情况下，只定位和修改上一轮（即最新轮）的答复中首个“不恰当 token”
- 由于你作为 LLM 只会输出文本，我们按照这个文本格式来输出你的 correcting 答复:
    - `{location_tokens}<SPLIT_TOKEN>{location_index}<SPLIT_TOKEN>{correcting_token}`
    - `{location_tokens}`: 用来定位 “修改位置” 的一串 tokens
        - 其内容为从不恰当的 token 开始，持续生成，直到触发以下任意情况：
            1. 在所有模型输出的 tokens 中 (包括模型的历史输出) 被 `{location_tokens}` 匹配上的第一处位置正好就是 “修改位置” 
                - 此时的 `{location_index}` 应该为 0。
                - 若第一匹配处不是 “修改位置”，则继续生成下一个 token 来做更加精准的定位
            2. `{location_tokens}` 长度达到 20 个 token 了，就该截止了。
                - 但是，若最后的几个 token 不能被你自己 (correcting model) 的 tokenizer decode 为完整字符，需要突破 20 tokens 限制生成到能 decode 出完整字符为止。
                - 若 20 个 token 都没法把 “修改位置” 准确定位，那就需要配合 `{location_index}` 来一起定位了。
            3. 一轮结束了，即已经生成了 stop token: <STOP_TOKEN>，也应该截止
    - <SPLIT_TOKEN> 是分隔内容的 special token
    - `{location_index}` 表示在所有模型输出的 tokens 中, 能被 `{location_tokens}` 匹配上的所有位置中的第几个位置
        - 是一个 int 数值，和 Python list 的 index 相似，从 0 开始计数。当用负数表示 index 时的绝对值比正数 index 更加小的时候，`{location_index}` 就用负数表示。
        - `{location_tokens}` 和 `{location_index}` 配合后，能在所有答复中共同定位一个唯一的位置，即 “第一个不恰当 token” 的位置
    - `{correcting_token}`: 更加恰当的 token，期望改为恰当 token 后，继续做补全能获得最好、最准确的答复。这里只需要一个 token 即可。
    -  stop token：上面的每一轮答复最后都有 stop token，需要的话，在 `{location_tokens}`,`{correcting_token}` 中使用 special token `<STOP_TOKEN>` 来表示 stop token
        - 比如, 要续写最后一轮的答复 `<STOP_TOKEN><SPLIT_TOKEN>-1<SPLIT_TOKEN>{continue token}`
    - tokenizer 问题：
        - 你需要通过多输出 token 或提前输出 token 来避免潜在的 tokenizer decode 出不合规文本的问题。
        - 即多个 tokens 对应一个文本字符的情况下，要把多个 token 视为一个整体，使所有输出的 tokens 能和文本互相转换，而不要截断中间 token
    - 如果 Correcting 范围内的回答都没有问题，输出一个 `<SPLIT_TOKEN>`

## example 1:
USER:
列举 3 种水果：
ASSISTANT:
苹果、土豆、香蕉
期望的输出: “土豆<SPLIT_TOKEN>0<SPLIT_TOKEN>西瓜”

## example 2:
USER:
Just reply 2 times, Using "|" as a separator：
1;2;3;4;5;6;7;8;9;8;
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;
USER:
Reply again
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;

期望的输出: “|1;2;3;4;5;6;7;8;9;8<SPLIT_TOKEN>-1<SPLIT_TOKEN><STOP_TOKEN>”
- “第一个不恰当 token”处和其他 ASSISTANT 的回答有重复，所以会生成完整个 20 个 `{location_tokens}`
- `{location_index}` 用正数表示时为 2， 用负数为 -1，其中， -1 绝对值更加小，所以应该用 -1
- 此处 `{correcting_token}` 为 stop token
"""

correcting_sft_system_prompt_default = correcting_sft_system_prompt_cn


class NextTokenPredictionAsLocationBuilder:
    def __init__(
        self,
        tokenizer=None,
        SPLIT_TOKEN="<SPLIT_TOKEN>",  # for qwen 2.5
        STOP_TOKEN="<STOP_TOKEN>",
        max_location_tokens=20,
    ):
        self.tokenizer = tokenizer or unicode_tokenizer
        self.SPLIT_TOKEN = SPLIT_TOKEN
        self.STOP_TOKEN = STOP_TOKEN
        self.max_location_tokens = max_location_tokens

    def get_correcting_sft_system_prompt(self, language="cn"):
        if language == "cn":
            prompt = correcting_sft_system_prompt_cn
        else:
            prompt = correcting_sft_system_prompt_default
        return prompt.replace("<SPLIT_TOKEN>", self.SPLIT_TOKEN).replace(
            "<STOP_TOKEN>", self.STOP_TOKEN
        )

    def convert_token_level_to_unicode_location(self, rejected_msgs):
        """
        根据 rejected_msgs 中的 token_level 信息返回 unicode_location

        Args:
            rejected_msgs: 消息列表

        Returns:
            dict: {"message_index": int, "unicode_location": int}
        """
        # 查找首个有 token_level 的 assistant 消息
        for i, msg in enumerate(rejected_msgs):
            if msg["role"] == "assistant" and "token_level" in msg:
                token_level = msg["token_level"]
                unicode_location = token_level["rejected_text_unicode_location"][0]
                return {"message_index": i, "unicode_location": unicode_location}

        raise ValueError("找不到包含 token_level 的 assistant 消息")

    def get_unicode_location(self, msgs, ntp_as_location):
        """
        Compute unicode_location by ntp_as_location in messages without token_level_info
        if Not found:
            return dict(not_found=True)
        """
        pass

    def messages_to_unicode_sequence(self, msgs, unicode_location=None):
        """
        Convert messages to a single text sequence, if unicode_location is given,
        also compute the sequence_location in the combined text sequence.

        Returns:
            update to unicode_location dict: {"assistant_sequence": str, "sequence_location": int (if unicode_location is given)}
        """

        # 收集所有assistant消息的内容，并记录其在原始消息中的索引
        assistant_contents = []
        assistant_indices = []
        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant":
                content = mxlm.get_text_content(msg["content"])
                # 添加隐藏的 STOP_TOKEN
                content += self.STOP_TOKEN
                content += "\n\n-----\n\n"
                assistant_contents.append(content)
                assistant_indices.append(i)

        assistant_sequence = "".join(assistant_contents)
        if unicode_location is None:
            unicode_location = {}
        else:
            message_index = unicode_location["message_index"]
            target_unicode_pos = unicode_location["unicode_location"]
            # 计算目标位置的unicode位置
            # 找到目标消息在assistant消息列表中的索引
            try:
                assistant_msg_idx = assistant_indices.index(message_index)
            except ValueError:
                raise ValueError(f"消息索引 {message_index} 不是 assistant 消息")

            current_pos = 0
            for i in range(assistant_msg_idx):
                current_pos += len(assistant_contents[i])
            sequence_location = current_pos + target_unicode_pos
            # unicode_location = deepcopy(unicode_location)
            unicode_location["sequence_location"] = sequence_location
        unicode_location["assistant_sequence"] = assistant_sequence
        # print(unicode_location)
        return unicode_location

    def set_location_index(self, rejected_msgs, ntp_as_location, unicode_location):
        """
        在所有模型输出的 tokens 中查找 ntp_as_location.location_string 的所有匹配位置，
        返回对应的索引位置 ntp_as_location.location_index

        Args:
            rejected_msgs: 消息列表
            ntp_as_location: dict(location_string=...) or 要查找的字符串
            unicode_location: dict, 包含 message_index 和 unicode_location, 也可以包含 assistant_sequence 和 sequence_location

        Returns ntp_as_location:
            int: location_index，从0开始计数，负数表示从末尾倒数
        """
        if isinstance(ntp_as_location, str):
            ntp_as_location = dict(location_string=ntp_as_location)
        ntp_as_location = deepcopy(ntp_as_location)
        location_string = ntp_as_location["location_string"]
        if "assistant_sequence" not in unicode_location:
            unicode_location = self.messages_to_unicode_sequence(
                rejected_msgs, unicode_location
            )
        assistant_sequence = unicode_location["assistant_sequence"]
        sequence_location = unicode_location["sequence_location"]

        # 在所有assistant内容中查找location_string的所有匹配位置
        matches = []
        start = 0
        while True:
            pos = assistant_sequence.find(location_string, start)
            if pos == -1:
                break
            matches.append(pos)
            start = pos + 1

        location_index = None
        # 找到目标位置对应的匹配索引
        for idx, match_pos in enumerate(matches):
            if match_pos == sequence_location:
                # 如果负数的绝对值更小，使用负数表示
                negative_idx = idx - len(matches)
                if abs(negative_idx) < idx:
                    location_index = negative_idx
                else:
                    location_index = idx

        ntp_as_location.update(
            unicode_location=unicode_location, matche_num=len(matches)
        )
        ntp_as_location["location_index"] = location_index
        if not len(matches):
            ntp_as_location["not_found"] = True
        return ntp_as_location

    def convert_rejected_content_to_ntp_as_location(self, rejected_msgs):
        """
        将 rejected_msgs 和 token_level_info 转换为 Next Token Prediction as location 格式

        Args:
            rejected_msgs: 消息列表

        Returns:
            dict: {"location_string": str, "location_index": int}
        """
        # 获取 unicode_location
        unicode_location = self.convert_token_level_to_unicode_location(rejected_msgs)
        message_index = unicode_location["message_index"]
        unicode_pos = unicode_location["unicode_location"]

        # 获取包含 token_level 的消息

        # 构建完整的 assistant 内容序列用于 tokenize
        assistant_contents = []
        assistant_indices = []
        for i, message in enumerate(rejected_msgs):
            if message["role"] == "assistant":
                content = mxlm.get_text_content(message["content"])
                # 添加隐藏的 STOP_TOKEN
                content += self.STOP_TOKEN
                assistant_contents.append(content)
                assistant_indices.append(i)

        full_content = "".join(assistant_contents)

        # tokenize 整个内容
        tokens = self.tokenizer.encode(full_content, add_special_tokens=False)

        # 计算目标位置在全部内容中的位置
        # 找到目标消息在assistant消息列表中的索引
        try:
            assistant_msg_idx = assistant_indices.index(message_index)
        except ValueError:
            raise ValueError(f"消息索引 {message_index} 不是 assistant 消息")

        current_pos = 0
        for i in range(assistant_msg_idx):
            current_pos += len(assistant_contents[i])
        target_global_pos = current_pos + unicode_pos

        # 找到对应的 token 位置
        # 通过逐步 decode 找到 unicode 位置对应的 token index
        token_idx = 0
        for i, token_id in enumerate(tokens):
            # decode 到当前位置的文本长度
            decoded_text = self.tokenizer.decode(
                tokens[: i + 1], skip_special_tokens=True
            )
            if len(decoded_text) > target_global_pos:
                token_idx = i
                break
            token_idx = i + 1

        # 使用 _minimal_reversible_patch 来获取合适的 token 范围
        start_idx, end_idx = _minimal_reversible_patch(
            tokens, token_idx, self.tokenizer
        )

        # 生成 location_tokens，限制在 max_location_tokens 内
        location_tokens_count = min(end_idx - start_idx, self.max_location_tokens)

        # 从 token_idx 开始生成 location_tokens，但要保证 tokenizer decode 的完整性
        location_start = token_idx
        location_end = token_idx + location_tokens_count

        # 确保 location_tokens 可以完整 decode
        while location_end <= len(tokens):
            try:
                test_tokens = tokens[location_start:location_end]
                test_text = self.tokenizer.decode(test_tokens, skip_special_tokens=True)
                # 重新 encode 检查是否一致
                reencoded = self.tokenizer.encode(test_text, add_special_tokens=False)
                if reencoded == test_tokens:
                    break
            except Exception:
                pass
            location_end += 1
            if (
                location_end - location_start > self.max_location_tokens + 10
            ):  # 避免无限循环
                break

        # 生成最终的 location_string
        location_tokens = tokens[location_start:location_end]
        location_string = self.tokenizer.decode(
            location_tokens, skip_special_tokens=True
        )

        # 如果 location_string 没有精确定位到目标位置，需要使用 location_index
        location_index = self.set_location_index(
            rejected_msgs, location_string, unicode_location
        )["location_index"]

        return {"location_string": location_string, "location_index": location_index}


if __name__ == "__main__":
    from boxx import *

    print("测试 NextTokenPredictionAsLocationBuilder 类的方法")

    # Example 1: 列举 3 种水果
    # USER: 列举 3 种水果：
    # ASSISTANT: 苹果、土豆、香蕉
    # 期望的输出: "土豆<SPLIT_TOKEN>0<SPLIT_TOKEN>西瓜"
    example1_msgs = [
        {"role": "user", "content": "列举 3 种水果："},
        {
            "role": "assistant",
            "content": "苹果、土豆、香蕉",
            "finish_reason": "stop",
            "token_level": {
                "chosen_text": "西瓜",
                "rejected_text": "土豆",
                "chosen_text_unicode_location": [3, 2],  # "土豆" 位于位置 3
                "rejected_text_unicode_location": [3, 2],
                "version": "1.0",
                "chosen_dialog_key": 2,
                "rejected_dialog_key": 1,
                "rejected_finish_reason": "stop",
            },
        },
    ]

    # Example 2: 多轮对话
    # USER: Just reply 2 times, Using "|" as a separator：1;2;3;4;5;6;7;8;9;8;
    # ASSISTANT: 1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;
    # USER: Reply again
    # ASSISTANT: 1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;
    # 期望的输出: "|1;2;3;4;5;6;7;8;9;8<SPLIT_TOKEN>-1<SPLIT_TOKEN><STOP_TOKEN>"
    example2_msgs = [
        {
            "role": "user",
            "content": 'Just reply 2 times, Using "|" as a separator：\n1;2;3;4;5;6;7;8;9;8;',
        },
        {
            "role": "assistant",
            "content": "1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;",
            "finish_reason": "stop",
        },
        {"role": "user", "content": "Reply again"},
        {
            "role": "assistant",
            "content": "1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;",
            "finish_reason": "stop",
            "token_level": {
                "chosen_text": "<STOP_TOKEN>",
                "rejected_text": "|1;2;3;4;5;6;7;8;9;8;",
                "chosen_text_unicode_location": [41, 1],
                "rejected_text_unicode_location": [41, 1],
                "version": "1.0",
                "chosen_dialog_key": 4,
                "rejected_dialog_key": 3,
                "rejected_finish_reason": "stop",
            },
        },
    ]

    # 创建 NextTokenPredictionAsLocationBuilder 实例
    builder = NextTokenPredictionAsLocationBuilder(tokenizer=unicode_tokenizer)

    # 测试基础方法
    print("\n=== 测试 Example 1 ===")
    unicode_location1 = builder.convert_token_level_to_unicode_location(example1_msgs)
    print(f"unicode_location: {unicode_location1}")

    location_index1 = builder.set_location_index(
        example1_msgs, "土豆", unicode_location1
    )
    print(f"location_index for '土豆': {location_index1}")

    print("\n=== 测试 Example 2 ===")
    unicode_location2 = builder.convert_token_level_to_unicode_location(example2_msgs)
    print(f"unicode_location: {unicode_location2}")

    location_index2 = builder.set_location_index(
        example2_msgs, "|1;2;3;4;5;6;7;8;9;8;", unicode_location2
    )
    print(f"location_index for '|1;2;3;4;5;6;7;8;9;8;': {location_index2}")

    print("\n基础方法测试完成")

    # 测试完整的转换方法
    try:
        print("\n=== 测试 convert_rejected_content_to_ntp_as_location ===")

        print("--- Example 1 ---")
        result1 = builder.convert_rejected_content_to_ntp_as_location(example1_msgs)
        print(
            f"Result: location_string='{result1['location_string']}', location_index={result1['location_index']}"
        )
        print("Expected format: '土豆<SPLIT_TOKEN>0<SPLIT_TOKEN>西瓜'")

        print("--- Example 2 ---")
        result2 = builder.convert_rejected_content_to_ntp_as_location(example2_msgs)
        print(
            f"Result: location_string='{result2['location_string']}', location_index={result2['location_index']}"
        )
        print(
            "Expected format: '|1;2;3;4;5;6;7;8;9;8<SPLIT_TOKEN>-1<SPLIT_TOKEN><STOP_TOKEN>'"
        )

        print("\n完整方法测试完成")

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
