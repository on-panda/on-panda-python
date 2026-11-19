"""Build a multimodal correcting-model trajectory without changing response templates."""

import json
from copy import deepcopy

import mximport
from mxlm.response_templates import CorrectionCallResponseTemplate

with mximport.inpkg():
    from .schema import correct_once_tool_cn, system_prompt_cn


def build_initial_cm_traj(
    policy_traj, base_cm_traj=None, extra_context=None, correction_scope="last_assistant"
):
    """Build the initial CM trajectory, marking the policy's correction scope in place.

    base_cm_traj can supply shared instructions, few-shot turns and other tools. It
    must not include correct_once; this builder registers it. extra_context is optional
    text specific to this policy trajectory.
    Adjacent text items are merged without crossing nontext content items.
    A missing assistant finish_reason still means partial output.
    """
    assert correction_scope in ("last_assistant", "after_last_user", "all"), correction_scope
    area_start = 0
    if correction_scope != "all":
        area_start = next(
            index
            for index in range(len(policy_traj["messages"]) - 1, -1, -1)
            if policy_traj["messages"][index]["role"]
            == ("assistant" if correction_scope == "last_assistant" else "user")
        )

    cm_traj = deepcopy(base_cm_traj) if base_cm_traj is not None else {}
    marker = "<correcting_prompt>"
    for message in cm_traj.setdefault("messages", []):
        if message["role"] != "system":
            continue
        if isinstance(message["content"], str):
            if marker not in message["content"]:
                message["content"] = marker + "\n" + message["content"]
        elif not any(
            item["type"] == "text" and marker in item["text"]
            for item in message["content"]
        ):
            message["content"].insert(0, dict(type="text", text=marker + "\n"))
    cm_traj["messages"].insert(
        0,
        dict(
            role="system",
            content=system_prompt_cn,
        ),
    )
    cm_traj.setdefault("tool_configs", []).append(
        dict(deepcopy(correct_once_tool_cn), require_approval="always")
    )
    if "tools" in cm_traj:
        cm_traj["tools"].append(deepcopy(correct_once_tool_cn))

    text = ""
    if extra_context is not None:
        text += f"{marker}\n## 本轮 Correcting 的补充信息\n{extra_context}\n\n"
    if policy_traj.get("tools"):
        text += (
            "### Tool Schemas for Policy Model Only\n```json\n"
            + json.dumps(policy_traj["tools"], ensure_ascii=False, indent=2)
            + "\n```\n这些工具只属于 policy model，供评估参考，不是让 correcting model 调用的工具。\n\n"
        )
    text += "### Policy Model Messages (rollout-1)\n"
    cm_content = []
    tool_names = {}
    response_template = CorrectionCallResponseTemplate()
    for index, message in enumerate(policy_traj["messages"]):
        correction = message.get("sidecar", {}).get("correction", {})
        correction.pop("_correction_area", None)
        if (
            message["role"] == "assistant"
            and index >= area_start
            and not correction.get("ignore")
        ):
            correction = message.setdefault("sidecar", {}).setdefault("correction", {})
            correction["_correction_area"] = True

        for call in message.get("tool_calls") or []:
            if call.get("id") and call.get("function", {}).get("name"):
                tool_names[call["id"]] = call["function"]["name"]
        if index:
            text += "\n__________\n"
        text += f'<{response_template.namespace}message role="{message["role"]}" index="{index + 1}"'
        if message["role"] == "tool":
            text += (
                f' name="{message.get("name") or tool_names[message["tool_call_id"]]}"'
                f' tool_call_id="{message["tool_call_id"]}"'
            )
        if correction.get("_correction_area"):
            text += ' correction="1"'
        text += ">\n"

        template_message = dict(message)
        if message["role"] != "assistant":
            template_message["finish_reason"] = "stop"
        if message.get("reasoning_content") is not None:
            template_message["reasoning"] = (
                message.get("reasoning") or message["reasoning_content"]
            )
        if isinstance(message.get("content"), list):
            template_message["content"] = ""
        applied = response_template.apply(template_message)
        if isinstance(message.get("content"), list):
            # Splice real content items into the empty mapped channel, not placeholders.
            content_start = next(
                mapping["text_start"]
                for mapping in applied["key_path_prompt_mapping"]
                if mapping["key_path"] == ["content"]
            )
            text += applied["templated_prompt"][:content_start]
            for item in message["content"]:
                if item["type"] == "text":
                    text += item["text"]
                else:
                    if text:
                        cm_content.append(dict(type="text", text=text))
                        text = ""
                    cm_content.append(deepcopy(item))
            text += applied["templated_prompt"][content_start:]
        else:
            text += applied["templated_prompt"]
    if text:
        cm_content.append(dict(type="text", text=text))
    cm_traj["messages"].append(dict(role="user", content=cm_content))
    return cm_traj


if __name__ == "__main__":
    from mxlm import AgentSession, ChatAPI
    from boxx import tree

    policy_traj = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "caption",
                    "description": "Submit a caption for the image.",
                    "parameters": {
                        "type": "object",
                        "properties": {"caption": {"type": "string"}},
                        "required": ["caption"],
                    },
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Call caption twice to describe this image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {  #  Image.new("RGB", (8, 8), "red")
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP8z4AdMOEQH6QSAM1BAQ/oQeJvAAAAAElFTkSuQmCC"
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "caption-1",
                        "type": "function",
                        "function": {
                            "name": "caption",
                            "arguments": '{"caption": "A red image."}',
                        },
                    },
                    {
                        "id": "caption-2",
                        "type": "function",
                        "function": {
                            "name": "caption",
                            "arguments": '{"caption": "It is a black image."}',
                        },
                    },
                ],
                "finish_reason": "tool_calls",
            },
        ],
    }
    cm_traj = build_initial_cm_traj(policy_traj)
    agent = AgentSession(ChatAPI.free_api(stream=False), cm_traj)
    loop_res = agent.run_agentic_loop()
    tree(loop_res)
    cm_response = loop_res["trajectory"]["messages"][-1]
    print(cm_response)
