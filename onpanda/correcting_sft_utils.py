#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 21:19:59 2025

@author: DIYer22
"""

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

# Next Token Prediction as location, return unicode_location dict(message_index=int, unicode_location=int)
# def get_unicode_location(rejected_msgs, ntp_as_location):  

# return location_index
# def get_location_index(rejected_msgs, location_string, unicode_location):


# convert msgs and token_level to Next Token Prediction as location
# def convert_rejected_content_to_ntp_as_location

if __name__ == "__main__":

    rejected_msgs_cts = [
        {
            "role": "user",
            "content": "写藏头诗：\n人工智能，大有可为",
        },
        {
            "role": "assistant",
            "ignore_loss": True,
            "content": "人智交融创新篇，  \n工巧技艺谱华年。  \n大展宏图前景阔，  \n有志竟成梦终圆。  \n可期未来科技盛，  \n为民造福永绵延。",
            "finish_reason": "stop",
            "token_level": {
                "chosen_text": "智",
                "rejected_text": "大",
                "chosen_text_unicode_location": [22, 1],
                "rejected_text_unicode_location": [22, 1],
                "version": "1.0",
                "chosen_dialog_key": 3,
                "rejected_dialog_key": 2,
                "rejected_finish_reason": "stop",
            },
        },
    ]
