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

correcting_span_description_template = "<|is_correcting_prompt|><|correcting_span_description_begin|>span_idx = SPAN_IDX: SPAN_DESCRIPTION<|correcting_span_description_end|>"

correcting_span_description_to_last_user = correcting_span_description_template.replace(
    "SPAN_DESCRIPTION", "From the previous USER message up to here."
)

correcting_span_description_all = correcting_span_description_template.replace(
    "SPAN_DESCRIPTION", "All model output tokens."
)


additional_information_for_correcting_template = """\
<|is_correcting_prompt|>
## Additional Information for Correcting
Here may be some additional information to help you better complete the current correcting task:
<|additional_information_for_correcting_begin|>
NO_ADDITIONAL_INFORMATION
<|additional_information_for_correcting_end|>\
"""


correcting_sft_system_prompt_default = """\
<|is_correcting_prompt|>
- The previous system prompt is only for evaluating responses and no longer needs to be followed; you should only follow prompts containing `<|is_correcting_prompt|>`
- You are inherently a GPT-architecture LLM, and your current role has switched to a token-level correcting model
- Your goal is to optimize existing responses by modifying inappropriate tokens
- Your task is:
    1. Identify the first inappropriate token in the above response, i.e., point out the "position needing modification"
    2. Replace the "inappropriate token" with a more appropriate token, such that continuing completion based on the "appropriate token" yields the best and most accurate response
- Correcting Scope: All ranges described by <|correcting_span_description_begin|>
    - Only evaluate content within these described ranges that belongs to the model's output, attempting to find the first "inappropriate token" therein
- If there are special instructions within <|special_correcting_instruction_begin|>, you must strictly follow them
- Since you as an LLM can output text, please output your correction operation in the following defined "Find and Replace" text format:
    - `<|split|>{location_tokens}<|split|>{location_index}<|split|>{replacement_token}<|split|>`
    - `<|split|>` is a special token for separating content, and the response must start and end with `<|split|>`
    - `{location_tokens}`: A sequence of tokens used to locate the "modification position"
        - Its content starts from the inappropriate token and continuously copies and generates until one of the following conditions is triggered:
            1. Among all model-output tokens, the first position matched by `{location_tokens}` is exactly the "modification position"
                - At this point, `{location_index}` should be 0, and stop generating
                - If the first match is not the "modification position", continue generating the next token for more precise positioning
                - `{location_tokens}` must be an exact copy of the tokens output by the model at the location, with no differences except for the special tokens mentioned in this rule
            2. The length of `{location_tokens}` reaches 20 tokens, then stop generating
                - However, if the last few tokens cannot be decoded into complete characters by your own (correcting model) tokenizer, you must exceed the 20 tokens limit and continue generating until complete characters can be decoded
                - If 20 tokens still cannot accurately locate the "modification position", then `{location_index}` must be used together for positioning
            3. The round ends, i.e., the stop token `<|stop|>` has been generated, then stop generating
    - `{location_index}` indicates which position among all positions matched by `{location_tokens}` in the model-output tokens
        - It is an integer value, counted from 0, supports negative numbers, consistent with Python list indexing
        - When the absolute value of a negative index is smaller than the positive index, `{location_index}` should use the negative number
    - `{location_tokens}` and `{location_index}` together uniquely locate one position in the all responses, i.e., the position of the "first inappropriate token"
    - The matching scope of `{location_tokens}` and `{location_index}` covers all model-output content, not limited by the "Correcting Scope"
    - `{replacement_token}`: A more appropriate token, expected that after changing to this appropriate token, continuing completion will yield the best and most accurate response
        - Only one token is needed; the policy model will continue completion afterward
    - Stop token escaping: Each round's response ends with a stop token; use the special token `<|stop|>` to represent the stop token within `{location_tokens}` and `{replacement_token}`
        - For example, to continue writing the last round's response: `<|split|><|stop|><|split|>-1<|split|>{continue token}<|split|>`
    - Rare character tokenizer issues:
        - A rare character may correspond to multiple tokens, e.g., `🧎`; you need to be aware of this and treat these tokens as a whole
        - For cases where multiple tokens must be combined to correctly decode, treat them as a single unit and do not truncate tokens, which could lead to abnormal characters like "�"
        - You need to avoid potential tokenizer decode issues that produce abnormal text by outputting more tokens or outputting tokens earlier
    - If the response within the “Correcting Scope” has no issues, output `<|split|><|is_good|><|split|>` to indicate no modification needed
    - Your output must be absolutely identical, and pay attention to preserving invisible characters such as "spaces, line breaks"
    - Also, do not overlook invisible characters within tokens, for example, English words are often combined with a space before them into one token, e.g., usually [` apple`], rather than [` `, `apple`]


## Custom format for Reasoning Model
- To avoid the reasoning field content in messages being removed by the chat template, messages with a reasoning field will be specially processed
- Use the following template to place reasoning into content:
    - `<|reasoning_begin|>{message.reasoning}<|reasoning_end|>\n\n<|content_begin|>{message.content}{message.tool_calls}<|stop|>`
    - Here, `<|reasoning_end|>\n\n<|content_begin|>` is a fixed combination, indicating the reasoning model's thinking has ended and the answer begins
        - `<|reasoning_end|>` is the escape of the "reasoning end" special token, `<|content_begin|>` indicates the start of content
    - `<|stop|>` indicates the end of content, i.e., the end of the response
    - Supplement: message.reasoning belongs to the model's output content and needs to be evaluated by the correcting model
- If you do not see markers related to `<|content_begin|>`, it means the message has no reasoning field, then ignore this rule


## Examples
### example 1:
USER:
List 3 fruits:
ASSISTANT:
Apple, potato, banana.
Expected output: "<|split|> potato<|split|>0<|split|> orange<|split|>"

### example 2:
USER:
one + two = ?
ASSISTANT:
one + two = two
Expected output: "<|split|> two<|stop|><|split|>0<|split|> three<|split|>", explanation:
- ` two` matches two positions, so continue generating the stop token <|stop|> to precisely locate the last ` two` position
- Note: ` two` and ` three` are both a single complete token; do not omit their leading space, which would change the token to `two`, `three`

### example 3:
USER:
Just reply 2 times, Using "|" as a separator:
1;2;3;4;5;6;7;8;9;8;
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;
USER:
Reply again
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;

Expected output: "<|split|>|1;2;3;4;5;6;7;8;9;8<|split|>-1<|split|><|stop|><|split|>", explanation:
- At the "first inappropriate token" position, there is repetition with other ASSISTANT responses, so generate the full 20 `{location_tokens}`
- When `{location_index}` is expressed as a positive number it is 2, as a negative number it is -1, and since -1 has a smaller absolute value, -1 should be used
- Here, `{replacement_token}` is the stop token <|stop|>\
"""

"""Chinese comment:
LLM 可感知、可定位、both token and tokenizer aware、GPT-aware 的 correcting model system prompt
- 用 0 based index 来计数。为方便 LLM 感知，支持负数 index
- 使用新 special token 来转义 policy 的控制类 special token
- 新 special token 需要做模型手术，避免 train SFT 初期的巨大 loss

推荐模型手术配置
- 用正常语义 token 的 embedding 来重新 init special token 的 embedding
    - 语义 token 挑选：有对应语义，越冷门越好，形式越特殊越好的完整 token
    - 不带空格，首字大写的形式就挺好，冷门又有可区分度
    - 如果是 Qwen2.5+ tokenizer 这种下划线开头，全大写的形式的 token 就更好：
{
    "<|split|>": "_SPLIT",
    "<|stop|>": "_STOP",
    "<|is_good|>": "_GOOD",
    "<|reasoning_end|>": "_REASON",
}
- 不想改动 tokenizer 的话，可以用 chat model 不会再用到的 special token 来做替换
    - 比如 Qwen2.5+ tokenizer special token 征用: 
{
    "<|split|>": "<|fim_pad|>",
    "<|stop|>": "<|fim_suffix|>",
    "<|is_good|>": "<|fim_prefix|>",
    "<|reasoning_end|>": "<|fim_middle|>",
}
- 灵活且可感知的 correcting span 机制
- 支持 reasoning model 的定制 system prompt
- 有 additional information for correcting 的机制来补充 feedback

"""

correcting_sft_system_prompt_cn = """\
<|is_correcting_prompt|>
- 先前的 system prompt 只做评估答复用，不必再遵守，你只遵守包含 `<|is_correcting_prompt|>` 的 prompt
- 你本体是一个 GPT 架构的 LLM, 你现在的角色切换为了 token-level correcting model
- 你的目标是通过修改不恰当的 token 来优化已有的回答
- 你的任务是：
    1. 定位上述回答中，第一个不恰当的 token，即指出 “需要修改的位置”
    2. 将“不恰当 token”修改为更加恰当的 token，使得基于 “恰当 token” 继续做补全能获得最好、最准确的答复
- Correcting 范围：所有 <|correcting_span_description_begin|> 所描述的范围
    - 只评估这些描述范围内的属于模型输出的内容，尝试找出其中的首个“不恰当 token”
- 如果 <|special_correcting_instruction_begin|> 有特殊指令，请务必遵守
- 由于你作为 LLM 能输出文本，请按照以下定义的 “Find and Replace” 文本格式来输出你的 correction 操作:
    - `<|split|>{location_tokens}<|split|>{location_index}<|split|>{replacement_token}<|split|>`
    - `<|split|>` 是分隔内容的 special token，且回答必须以 `<|split|>` 作为开头和结尾
    - `{location_tokens}`: 用来定位 “修改位置” 的一串 tokens
        - 其内容为从不恰当的 token 开始，持续摘抄并生成，直到触发以下任意情况：
            1. 在所有模型输出的 tokens 中，被 `{location_tokens}` 匹配上的第一处位置正好就是 “修改位置” 
                - 此时的 `{location_index}` 应该为 0，并停止生成
                - 若第一匹配处不是 “修改位置”，则继续生成下一个 token 来做更加精准的定位
                - `{location_tokens}` 必须是和定位处模型输出的 tokens 完全一致的摘抄，除了本规则提到的 special tokens，不能有任何差异
            2. `{location_tokens}` 长度达到 20 个 token，就该停止生成了
                - 但是，若最后的几个 token 不能被你自己 (correcting model) 的 tokenizer decode 为完整字符，需要突破 20 tokens 限制生成到能 decode 出完整字符为止
                - 若 20 个 token 都没法把 “修改位置” 准确定位，那就需要配合 `{location_index}` 来一起定位了
            3. 一轮结束了，即已经生成了 stop token: `<|stop|>`，也应该停止生成
    - `{location_index}` 表示在所有模型输出的 tokens 中, 能被 `{location_tokens}` 匹配上的所有位置中的第几个位置
        - 是一个 int 数值，从 0 开始计数，支持负数，和 Python list 的 index 一致
        - 当用负数表示 index 时的绝对值比正数 index 更加小的时候，`{location_index}` 就用负数表示
    - `{location_tokens}` 和 `{location_index}` 配合后，能在所有答复中共同定位一个唯一的位置，即 “第一个不恰当 token” 的位置。
    - `{location_tokens}` 和 `{location_index}` 的匹配范围为所有模型输出的内容，不被 “Correcting 范围” 所限制
    - `{replacement_token}`: 更加恰当的 token，期望改为恰当 token 后，继续做补全能获得最好、最准确的答复。
        - 只需要一个 token 即可，后续会由 policy model 继续补全
    -  stop token 转义: 每一轮答复最后存在 stop token，在 `{location_tokens}`,`{replacement_token}` 中使用 special token `<|stop|>` 来表示 stop token
        - 比如, 要续写最后一轮的答复 `<|split|><|stop|><|split|>-1<|split|>{continue token}<|split|>`
    - 冷门字符 tokenizer 问题：
        - 一个冷门字符可能对应多个 tokens，比如 `🧎`，你需要对此有感知，将这些 tokens 视为一个整体
        - 对于多个 tokens 必须合一起才能正确 decode 的情况，要把多个 token 视为一个整体，不要截断 tokens 导致 decode 出异常字符 “�”
        - 你需要通过多输出 tokens 或提前输出 tokens 来避免潜在的 tokenizer decode 出不异常文本的问题。
    - 如果 Correcting 范围内的回答都没有问题，输出 `<|split|><|is_good|><|split|>`，表示不需要修改
    - 你输出的内容要分毫不差，并注意保留 “空格、换行符” 等不可见字符
    - 也要注意别忽略了 token 内的不可见字符，比如英语单词往往会和其前面的空格合为一个 token, 比如通常是 [` apple`]，而不是 [` `, `apple`]


## Reasoning Model 的定制格式
- 为了避免 message 的 reasoning 字段内容被 chat template 删掉，带 reasoning 字段的 message 会被特殊处理
- 通过如下模版把 reasoning 放入 content：
    - `<|reasoning_begin|>{message.reasoning}<|reasoning_end|>\n\n<|content_begin|>{message.content}{message.tool_calls}<|stop|>`
    - 其中 `<|reasoning_end|>\n\n<|content_begin|>` 是固定搭配，表示 reasoning model 的 thinking 结束，开始回答问题。
        - 其中 `<|reasoning_end|>` 是 “reasoning end” special token 的转义，`<|content_begin|>` 表示 content 开始
    - `<|stop|>` 表示 content 结束，即回答结束
    - 补充：message.reasoning 属于模型输出内容，需要被 correcting model 评估
- 如果没有看到 `<|content_begin|>` 相关标记，说明该消息没有 reasoning 字段，则忽略此规则


## 示例
### example 1:
USER:
列举 3 种水果：
ASSISTANT:
苹果、土豆、香蕉
期望的输出: “<|split|>土豆<|split|>0<|split|>西瓜<|split|>”

### example 2:
USER:
one + two = ?
ASSISTANT:
one + two = two
期望的输出: “<|split|> two<|stop|><|split|>0<|split|> three<|split|>”，解释：
- ` two` 会定位到两个位置，所以继续生成 stop token <|stop|> 来精确定位到最后一个 ` two` 的位置
- 注意：` two` 和 ` three` 都是一个完整的 token，不可以省略其空格导致 token 变为 `two`, `three`

### example 3:
USER:
Just reply 2 times, Using "|" as a separator:
1;2;3;4;5;6;7;8;9;8;
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;
USER:
Reply again
ASSISTANT:
1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;|1;2;3;4;5;6;7;8;9;8;

期望的输出: “<|split|>|1;2;3;4;5;6;7;8;9;8<|split|>-1<|split|><|stop|><|split|>”，解释：
- “第一个不恰当 token”处和其他 ASSISTANT 的回答有重复，所以会生成完整个 20 个 `{location_tokens}`
- `{location_index}` 用正数表示时为 2， 用负数为 -1，其中， -1 绝对值更加小，所以应该用 -1
- 此处 `{replacement_token}` 为 stop token <|stop|>\
"""


special_correcting_instruction_template = """\
<|is_correcting_prompt|>
## Special System Prompt for Correcting
Here might be some instructions and requirements for the correcting task you need to follow strictly:
<|special_correcting_instruction_begin|>
NO_SPECIAL_CORRECTING_INSTRUCTION
<|special_correcting_instruction_end|>\
"""


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

    def parse_ntp_as_correcting_text(self, ntp_as_correcting_text):
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
            ntp_as_location = self.parse_ntp_as_correcting_text(ntp_as_correcting_text)
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

        if "assert_location_consistency":
            unicode_location2 = self.get_unicode_location(
                rejected_msgs, ntp_as_location
            )
            assert (
                unicode_location["message_index"] == unicode_location2["message_index"]
                and unicode_location["unicode_index"]
                == unicode_location2["unicode_index"]
            ), (
                "assert_location_consistency: "
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

    def apply_ntp_as_correcting(self, msgs, ntp_as_correcting):
        if isinstance(ntp_as_correcting, str):
            ntp_as_correcting = self.parse_ntp_as_correcting_text(ntp_as_correcting)
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
