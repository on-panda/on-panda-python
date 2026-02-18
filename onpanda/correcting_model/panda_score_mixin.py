#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 04:52:00 2026

@author: DIYer22
"""
import os
import mximport

with mximport.inpkg():
    from ..utils import remove_msgs_after_last_response_role

far_tokenizer_agnostic_system_prompt_default = """\
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


far_tokenizer_agnostic_system_prompt_cn = """\
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

class PandaScoreMixin:
    @staticmethod
    def _mean(nums):
        return sum(nums) / len(nums)

    def eval_panda_score(self, panda_json_paths):
        """
        Evaluate correcting quality on panda json files.

        Returned summary metrics:
        - format_score: mean format reward on all samples
        - location_score: mean location reward on all samples
        - replacement_score: mean replacement reward on all samples
        - is_good_cls_score: accuracy of predicting is_good on all samples
        - location_on_not_good_score: mean location reward on not_good samples
        - replacement_on_not_good_score: mean replacement reward on not_good samples
        - replacement_on_true_location_score: mean replacement reward on not_good
          samples where location is correct
        - not_good_num: number of not_good samples
        - true_location_num: number of not_good samples with correct location
        """
        from onpanda import PandaTree
    
        if isinstance(panda_json_paths, str):
            panda_json_paths = [panda_json_paths]

        format_scores = []
        location_scores = []
        replacement_scores = []
        is_good_cls_scores = []
        location_on_not_good_scores = []
        replacement_on_not_good_scores = []
        replacement_on_true_location_scores = []
        not_good_num = 0
        true_location_num = 0
        score_results = {}

        for json_path in panda_json_paths:
            panda_tree = PandaTree(json_path, tokenizer=self.adapter.tokenizer)
            far_correction_datas = panda_tree.build_far_correction_data_v1(self.adapter)
            for far_correction_data_idx, far_correction_data in enumerate(
                far_correction_datas
            ):
                messages = remove_msgs_after_last_response_role(
                    far_correction_data[:-2]
                )
                gt_correction = far_correction_data[-1]["correcting"]

                correcting_result = self.correcting(messages)
                far_text = correcting_result["far_text"]
                reward_result = self.adapter.verifier.compute_reward(
                    messages,
                    far_text,
                    gt_correction,
                )
                reward_with_feedback = reward_result["reward_with_feedback"]

                gt_is_good = bool(
                    gt_correction["find_and_replace"].get("is_good")
                    or gt_correction["messages_location"].get("is_good")
                )
                pred_is_good = bool(correcting_result.get("is_good"))
                panda_score = dict(
                    format_score=reward_with_feedback["format_reward"],
                    location_score=reward_with_feedback["location_reward"],
                    replacement_score=reward_with_feedback["replacement_reward"],
                    is_good_cls_score=1.0 if gt_is_good == pred_is_good else 0.0,
                )
                reward_result["panda_score"] = panda_score
                score_results[(json_path, far_correction_data_idx)] = reward_result

                format_scores.append(panda_score["format_score"])
                location_scores.append(panda_score["location_score"])
                replacement_scores.append(panda_score["replacement_score"])
                is_good_cls_scores.append(panda_score["is_good_cls_score"])

                if not gt_is_good:
                    not_good_num += 1
                    location_on_not_good_scores.append(panda_score["location_score"])
                    replacement_on_not_good_scores.append(
                        panda_score["replacement_score"]
                    )
                    if panda_score["location_score"] == 1.0:
                        true_location_num += 1
                        replacement_on_true_location_scores.append(
                            panda_score["replacement_score"]
                        )

        return dict(
            format_score=self._mean(format_scores),
            location_score=self._mean(location_scores),
            replacement_score=self._mean(replacement_scores),
            is_good_cls_score=self._mean(is_good_cls_scores),
            location_on_not_good_score=(
                self._mean(location_on_not_good_scores) if not_good_num else 0.0
            ),
            replacement_on_not_good_score=(
                self._mean(replacement_on_not_good_scores) if not_good_num else 0.0
            ),
            replacement_on_true_location_score=(
                self._mean(replacement_on_true_location_scores)
                if true_location_num
                else 0.0
            ),
            not_good_num=not_good_num,
            true_location_num=true_location_num,
            score_results=score_results,
        )


if __name__ == "__main__":
    from boxx import tree
    from glob import glob

    with mximport.inpkg():
        from .correcting_model import build_test_correcting_model

    correcting_model = build_test_correcting_model(tokenizer=None)
    test_json_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "../../../on-panda-example-data/panda_json/2025-04-12_Chinese_acrostic_poem_藏头诗_tokenizer-step2.panda.json",
        )
    ]

    # correcting_model = build_test_correcting_model(tokenizer="/home/yl/audio/asset/tokenizer/step1f/")
    # test_json_paths = sorted(glob("/home/yl/audio/jili_tracelogs_bmk/step1f_tracelogs2_batch5_jili_bmk/*.panda.json"))[:3]
    panda_scores = correcting_model.eval_panda_score(test_json_paths)
    print(
        {
            k: panda_scores[k]
            for k in (
                "format_score",
                "location_score",
                "replacement_score",
                "is_good_cls_score",
                "location_on_not_good_score",
                "replacement_on_not_good_score",
                "replacement_on_true_location_score",
                "not_good_num",
                "true_location_num",
            )
        }
    )
    score_results = list(panda_scores["score_results"].values())
    tree(score_results[-1])
