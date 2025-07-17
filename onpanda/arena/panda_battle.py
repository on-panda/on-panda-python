import random

judge_message_default_cn = dict(
    role="system",
    content="现在你的身份变成了一个评判者，要对答复进行对比和评价。上面的最后一轮有上下两条答复，请分析和对比两条答复在对应语境和设定下的质量，最后选出一个最佳答复，允许平手。分析和评价完成后，需要把最优的答复以 `answer{}` 格式包裹，只允许填写 上、平、下， 比如 `answer{平}`",
)
custom_single_choice_default_cn = {
    "name": "谁更好",
    "type": "single_choice",
    "single_choice": [
        {"k": "上", "v": None},
        {"k": "平", "v": None},
        {"k": "下", "v": None},
    ],
    "tips": "最后一轮有上下两条答复, 谁更好? 允许平手",
    "required": True,
}


def build_panda_battle(
    arena_json1, arena_json2, judge_message=None, custom_single_choice=None
):
    if judge_message is None:
        judge_message = judge_message_default_cn
    if custom_single_choice is None:
        custom_single_choice = custom_single_choice_default_cn
    if "data" in arena_json1 and isinstance(arena_json1["data"], dict):
        arena_json1 = arena_json1["data"]
    if "data" in arena_json2 and isinstance(arena_json2["data"], dict):
        arena_json2 = arena_json2["data"]
    panda_list = []
    for prompt_key in arena_json1:
        if prompt_key not in arena_json2:
            continue
        msgs1 = arena_json1[prompt_key]
        msgs2 = arena_json2[prompt_key]
        is_switched = random.random() < 0.5
        if is_switched:
            up, down = msgs2[-1], msgs1[-1]
            exp_up = up["arena"]["exp_name"]
            exp_down = down["arena"]["exp_name"]
        else:
            up, down = msgs1[-1], msgs2[-1]
            exp_up = up["arena"]["exp_name"]
            exp_down = down["arena"]["exp_name"]

        hidden_info = dict(
            order=[exp_up, exp_down], is_switched=is_switched, prompt_key=prompt_key
        )
        judge_messages = msgs1[:-1] + [up, down, judge_message]
        panda = {
            "dialogs": {
                "1": {
                    "hidden_info": hidden_info,
                    "messages": judge_messages,
                    "annotate": {
                        "is_good": None,
                        "customs": [custom_single_choice],
                    },
                }
            },
            "description": f"prompt_name:{msgs1[-1]['arena'].get('prompt_name', prompt_key)}",
        }
        panda_list.append(panda)
    return panda_list


if __name__ == "__main__":
    pass
