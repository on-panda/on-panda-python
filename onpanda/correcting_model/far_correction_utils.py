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


far_correction_system_prompt_default = """\
<|is_correcting_prompt|>
- The previous system prompt is only for evaluating responses and no longer needs to be followed; you should only follow prompts containing `<|is_correcting_prompt|>`
- You are inherently a GPT-architecture LLM, and your current role has switched to a token-level correcting model
- Your goal is to optimize existing responses by modifying inappropriate tokens
- Your task is:
    1. Identify the first inappropriate token in the above response, i.e., point out the "position needing modification"
    2. Replace the "inappropriate token" with a more appropriate token, such that continuing completion based on the "appropriate token" yields the best and most accurate response
- Correcting Scope: All spans described by <|correcting_span_description_begin|>
    - Only evaluate content within these described spans that belongs to the model's output, attempting to find the first "inappropriate token" therein
    - If there is no span description starting with <|correcting_span_description_begin|>, evaluate the most recent response by default
- If there are special instructions within <|special_correcting_instruction_begin|>, you must strictly follow them
- Since you as an LLM can output text, please output your correction operation in the following defined "Find and Replace" text format:
    - `<|split|>{location_tokens}<|split|>{location_index}<|split|>{replacement_token}<|split|>`
    - `<|split|>` is a special token for separating content, and your response must start and end with `<|split|>`
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
    - `{message.reasoning}<|reasoning|>


<|reasoning|>{message.content}{message.tool_calls}<|stop|>`
    - Here, `<|reasoning|>


<|reasoning|>` is a fixed combination, indicating the reasoning model's thinking has ended and the answer begins
        - `<|reasoning|>` is the escape of the "thinking end" special token
    - `<|stop|>` indicates the end of content, i.e., the end of the response
    - {message.reasoning} belongs to the model's output content and needs to be evaluated by the correcting model
- If you do not see `<|reasoning|>` marker, it means the message has no reasoning field, then ignore this rule


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
    "<|reasoning|>": "_REASON",
}
- 不想改动 tokenizer 的话，可以用 chat model 不会再用到的 special token 来做替换
    - 比如 Qwen2.5+ tokenizer special token 征用: 
{
    "<|split|>": "<|fim_pad|>",
    "<|stop|>": "<|fim_suffix|>",
    "<|is_good|>": "<|fim_prefix|>",
    "<|reasoning|>": "<|fim_middle|>",
}
- 灵活且可感知的 correcting span 机制
- 支持 reasoning model 的定制 system prompt
- 有 additional information for correcting 的机制来补充 feedback

"""

far_correction_system_prompt_cn = """\
<|is_correcting_prompt|>
- 先前的 system prompt 只做评估答复用，不必再遵守，你只遵守包含 `<|is_correcting_prompt|>` 的 prompt
- 你本体是一个 GPT 架构的 LLM, 你现在的角色切换为了 token-level correcting model
- 你的目标是通过修改不恰当的 token 来优化已有的回答
- 你的任务是：
    1. 定位上述回答中，第一个不恰当的 token，即指出 “需要修改的位置”
    2. 将“不恰当 token”修改为更加恰当的 token，使得基于 “恰当 token” 继续做补全能获得最好、最准确的答复
- Correcting 范围：所有 <|correcting_span_description_begin|> 所描述的范围
    - 只评估这些描述范围内的属于模型输出的内容，尝试找出其中的首个“不恰当 token”
    - 若没有带 <|correcting_span_description_begin|> 的范围描述，则默认评估最后一条回答内容
- 如果 <|special_correcting_instruction_begin|> 有特殊指令，请务必遵守
- 由于你作为 LLM 能输出文本，请按照以下定义的 “Find and Replace” 文本格式来输出你的 correction 操作:
    - `<|split|>{location_tokens}<|split|>{location_index}<|split|>{replacement_token}<|split|>`
    - `<|split|>` 是分隔内容的 special token，且你的新回答必须以 `<|split|>` 作为开头和结尾
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
    - `{message.reasoning}<|reasoning|>


<|reasoning|>{message.content}{message.tool_calls}<|stop|>`
    - 其中 `<|reasoning|>


<|reasoning|>` 是固定搭配，表示 reasoning model 的 thinking 结束，开始正式回答问题。
        - `<|reasoning|>` 是 “thinking end” special token 的转义
    - `<|stop|>` 表示 content 结束，即回答结束
    - {message.reasoning} 属于模型输出内容，需要被 correcting model 评估
- 如果没有看到 `<|reasoning|>` 标记，说明该消息没有 reasoning 字段，则忽略此规则


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


class CorrectionAdapter:
    pass


class FindAndReplaceCorrectionAdapter(CorrectionAdapter):
    def __init__(
        self,
        tokenizer=None,
        special_tokens=None,
        max_location_tokens=20,
        scope_slice=(-1, None),
    ):
        self.tokenizer = tokenizer or unicode_tokenizer
        special_tokens = special_tokens or {}
        self.SPLIT_TOKEN = special_tokens.get("split", "<|split|>")
        self.STOP_TOKEN = special_tokens.get("stop", "<|stop|>")
        self.IS_GOOD_TOKEN = special_tokens.get("is_good", "<|is_good|>")
        self.REASONING_TOKEN = special_tokens.get("reasoning", "<|reasoning|>")
        self.max_location_tokens = max_location_tokens
        self.scope_slice = scope_slice

    def build_correction_prompt(self, messages, language=None):
        if language == "cn":
            prompt = far_correction_system_prompt_cn
        else:
            prompt = far_correction_system_prompt_default
        system_prompt = (
            prompt.replace("<|split|>", self.SPLIT_TOKEN)
            .replace("<|stop|>", self.STOP_TOKEN)
            .replace("<|is_good|>", self.IS_GOOD_TOKEN)
            .replace("<|reasoning|>", self.REASONING_TOKEN)
            .replace(" 20 ", f" {self.max_location_tokens} ")
        )
        sys_prompt_message = dict(
            role="system",
            content=system_prompt,
        )
        return messages + [sys_prompt_message]

    def parse(self, far_text):
        mid_text = far_text.removeprefix(self.SPLIT_TOKEN).removesuffix(
            self.SPLIT_TOKEN
        )
        # TODO: remove workaround for old step1f correcting model
        if mid_text == self.IS_GOOD_TOKEN or mid_text in ["", self.SPLIT_TOKEN]:
            find_and_replace = dict(
                is_good=True,
                location_text="",
                location_index=0,
                replacement_token="",
            )
        else:
            splits = mid_text.split(self.SPLIT_TOKEN)
            assert len(splits) == 3, far_text
            find_and_replace = dict(
                is_good=False,
                location_text=splits[0],
                location_index=int(splits[1]),
                replacement_token=splits[2],
            )
        return find_and_replace

    def _iter_assistant_text_locations(self, messages):
        for message_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str):
                yield [message_index, "reasoning"], reasoning

            content = message.get("content", "")
            if isinstance(content, list):
                assert all([d["type"] == "text" for d in content]), message
                content = mxlm.get_text_content(content)
            else:
                content = mxlm.get_text_content(content)
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

        location_index = find_and_replace["location_index"]
        location_text = find_and_replace.get("location_text", "")
        assert location_text, find_and_replace
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

    def _get_by_path(self, data, path_keys):
        target = data
        for key in path_keys:
            target = target[key]
        return target

    def _set_by_path(self, data, path_keys, value):
        target = data
        for key in path_keys[:-1]:
            target = target[key]
        target[path_keys[-1]] = value

    def convert_token_level_to_messages_location(self, rejected_messages):
        """
        根据 rejected_messages 中的 token_level 信息返回 messages_location
        """
        for message_index, message in enumerate(rejected_messages):
            if message["role"] == "assistant" and "token_level" in message:
                token_level = message["token_level"]
                if "messages_location" in token_level:
                    return deepcopy(token_level["messages_location"])
                char_index = token_level["rejected_text_unicode_range"][0]
                patch_length = token_level["rejected_text_unicode_range"][1]
                content = mxlm.get_text_content(message["content"])
                return dict(
                    path_keys=[message_index, "content"],
                    char_index=char_index,
                    patch_length=patch_length,
                    left5=content[max(0, char_index - 5) : char_index],
                    right5=content[char_index : char_index + 5],
                )
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
        for path_keys, text in self._iter_assistant_text_locations(rejected_messages):
            search_scope = text + self.STOP_TOKEN
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
        content = self._get_by_path(rejected_messages, path_keys)
        if isinstance(content, list):
            assert all([d["type"] == "text" for d in content]), rejected_messages
            content = mxlm.get_text_content(content)
        content_suffix = content[char_index:] + self.STOP_TOKEN
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
            messages_location2 = self.locate(rejected_messages, find_and_replace)
            assert (
                messages_location["path_keys"] == messages_location2["path_keys"]
                and messages_location["char_index"] == messages_location2["char_index"]
            ), (
                "assert_location_consistency: "
                + str(messages_location)
                + str(messages_location2)
                + str(find_and_replace)
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
            )
            is_good_correcting_msg = dict(
                role="assistant",
                content=f"{self.SPLIT_TOKEN}{self.IS_GOOD_TOKEN}{self.SPLIT_TOKEN}",
                correcting=dict(
                    messages_location=is_good_messages_location,
                    find_and_replace=is_good_find_and_replace,
                    scope_slice=self.scope_slice,
                ),
            )
            return self.build_correction_prompt(messages) + [is_good_correcting_msg]

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
        replacement_token = token_level_info["chosen_text"] or self.STOP_TOKEN
        find_and_replace.update(
            replacement_token=replacement_token,
            is_good=False,
        )
        correcting_content = (
            f"{self.SPLIT_TOKEN}{find_and_replace['location_text']}{self.SPLIT_TOKEN}"
            f"{find_and_replace['location_index']}{self.SPLIT_TOKEN}"
            f"{find_and_replace['replacement_token']}{self.SPLIT_TOKEN}"
        )
        correcting_msg = dict(
            role="assistant",
            content=correcting_content,
            correcting=dict(
                messages_location=messages_location,
                find_and_replace=find_and_replace,
                scope_slice=self.scope_slice,
            ),
        )
        far_correction = self.build_correction_prompt(rejected_messages) + [
            correcting_msg
        ]
        return far_correction

    def apply(self, messages, correction_or_far_text):
        if isinstance(correction_or_far_text, str):
            find_and_replace = self.parse(correction_or_far_text)
            correction = dict(find_and_replace=find_and_replace)
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
                raise AssertionError(correction)

        if "messages_location" not in correction:
            correction["messages_location"] = self.locate(messages, find_and_replace)

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
        partial_messages = deepcopy(messages[: path_keys[0] + 1])

        field_text = self._get_by_path(partial_messages, path_keys)
        if isinstance(field_text, list):
            assert all([d["type"] == "text" for d in field_text]), partial_messages
            field_text = mxlm.get_text_content(field_text)

        good_prefix = field_text[:char_index]
        if self.STOP_TOKEN == replacement_token:
            corrected_field_text = good_prefix
        else:
            corrected_field_text = good_prefix + replacement_token
        self._set_by_path(partial_messages, path_keys, corrected_field_text)

        if path_keys[-1] == "content":
            if self.STOP_TOKEN == replacement_token:
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
    rejected_msgs1, ntp_as_correcting_text_gt1 = get_test_rejected_msgs1()[:2]

    result1 = far_adapter.build_correction_from_rejected_messages(rejected_msgs1)
    assert result1["find_and_replace"]["location_text"] == " potato", result1
    assert result1["find_and_replace"]["location_index"] == 0, result1

    correction1 = far_adapter.apply(rejected_msgs1, ntp_as_correcting_text_gt1)
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
    ntp_as_correcting_text_gt2 = "<|fim_pad|>|1;2;3;4;5;6;7;8;9;8<|fim_pad|>-1<|fim_pad|><|fim_suffix|><|fim_pad|>"
    assert correcting_content2 == ntp_as_correcting_text_gt2, correcting_content2
    correction2 = far_adapter.apply(far_correction2[:-2], ntp_as_correcting_text_gt2)
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
