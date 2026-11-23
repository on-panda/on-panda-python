"""Render policy dialogs for the correcting model and apply a correct_once decision."""

import json
from copy import deepcopy
from os.path import commonprefix

import mximport
from mxlm.response_templates import (
    CorrectionCallResponseTemplate,
    build_messages_location,
)

with mximport.inpkg():
    from .schema import correct_once_tool_cn, system_prompt_cn


response_template = CorrectionCallResponseTemplate()


def build_initial_cm_traj(
    policy_traj,
    base_cm_traj=None,
    extra_context=None,
    correction_scope="last_assistant",
):
    """Build the initial CM trajectory for dialog 1, marking its correction scope in place.

    base_cm_traj can supply shared instructions, few-shot turns and other tools. It
    must not include correct_once; this builder registers it. extra_context is optional
    text specific to this policy trajectory, after the complete policy context.
    """

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
    if policy_traj.get("tools"):
        text += (
            "### Tool Schemas for Policy Model Only\n```json\n"
            + json.dumps(policy_traj["tools"], ensure_ascii=False, indent=2)
            + "\n```\n这些工具只属于 policy model，供评估参考，不是让 correcting model 调用的工具。\n\n"
        )
    cm_content = render_dialog(policy_traj, 1, [], correction_scope)
    cm_content[0]["text"] = text + "### Policy Model Messages\n" + cm_content[0]["text"]
    if extra_context is not None:
        cm_content[-1][
            "text"
        ] += f"\n\n{marker}\n## 本轮 Correcting 的补充信息\n{extra_context}\n</correcting_prompt>"
    cm_traj["messages"].append(dict(role="user", content=cm_content))
    return cm_traj


def _render_body(message, index):
    """Use one template space for both CM display and anchor lookup."""
    assert (
        "reasoning_content" not in message
    ), "Convert reasoning_content to reasoning before correction"
    # Every candidate is displayed as complete, including a length-limited response.
    template_message = dict(message, finish_reason="stop")
    if message["role"] != "assistant" and isinstance(message.get("content"), list):
        template_message["content"] = ""
    applied = response_template.apply(template_message)
    # mxlm anchors a body without mapped channels at the start of empty content.
    template_message.setdefault("content", "")
    return dict(applied, message=template_message, message_index=index)


def policy_prompt_length(messages):
    """Only trailing assistant messages are outside the shared policy prompt."""
    for i in reversed(range(len(messages))):
        if messages[i]["role"] != "assistant":
            return i + 1
    return 0


def render_dialog(
    policy_traj,
    dialog_key,
    ancestor_keys,
    correction_scope,
    *,
    last_turn_only=False,
    rejected_message=None,
):
    """Mark the full snapshot; a rejected message renders the final assistant as a delta.

    last_assistant covers the last consecutive group of assistant messages.
    A scope of None displays shared context without marking any message correctable.
    """
    messages = policy_traj["messages"]
    assert correction_scope in (
        "last_assistant",
        "after_last_user",
        "all",
        None,
    ), correction_scope
    area_start = len(messages) if correction_scope is None else 0
    if correction_scope in ("last_assistant", "after_last_user"):
        role = "assistant" if correction_scope == "last_assistant" else "user"
        area_start = next(
            i for i in reversed(range(len(messages))) if messages[i]["role"] == role
        )
        if correction_scope == "last_assistant":
            while area_start and messages[area_start - 1]["role"] == "assistant":
                area_start -= 1
    namespace = response_template.namespace
    dialog = f'<{namespace}dialog key="{dialog_key}" ancestor_keys="{json.dumps(ancestor_keys)}"'
    start = 0
    if last_turn_only or rejected_message is not None:
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(
                "A partial dialog must end with the changed assistant message"
            )
        dialog += ' partial_dialog="true"'
        start = policy_prompt_length(messages) if last_turn_only else len(messages) - 1
    parts = [dialog + ">\n"]
    tool_names = {}
    for index, message in enumerate(messages):
        correction = message.get("sidecar", {}).get("correction", {})
        correction.pop("_correction_area", None)
        if (
            message["role"] == "assistant"
            and index >= area_start
            and not correction.get("ignore")
        ):
            correction = message.setdefault("sidecar", {}).setdefault("correction", {})
            correction["_correction_area"] = True
        if index < start:
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id") and call.get("function", {}).get("name"):
                tool_names[call["id"]] = call["function"]["name"]
        if index != start:
            parts.append("\n__________\n")
        attrs = f'role="{message["role"]}" index="{index + 1}"'
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            name = message.get("name") or tool_names.get(call_id)
            if name is not None:
                attrs += f' name="{name}"'
            if call_id is not None:
                attrs += f' tool_call_id="{call_id}"'
        if correction.get("_correction_area"):
            attrs += ' correction="1"'
        body = _render_body(message, index)
        text = body["templated_prompt"]
        if rejected_message is not None:
            rejected = _render_body(rejected_message, index)["templated_prompt"]
            prefix = commonprefix((rejected, text))
            tail = prefix.encode("utf-8")[-50:].decode("utf-8", errors="ignore")
            parts.extend(
                (
                    f'<{namespace}message_delta {attrs} common_prefix_chars="{len(prefix)}">\n',
                    tail,
                    f"<{namespace}fork/>",
                    text[len(prefix) :],
                    f"\n</{namespace}message_delta>",
                )
            )
            continue
        parts.append(f"<{namespace}message {attrs}>\n")
        if message["role"] != "assistant" and isinstance(message.get("content"), list):
            offset = next(
                m["text_start"]
                for m in body["key_path_prompt_mapping"]
                if m["key_path"] == ["content"]
            )
            parts.append(text[:offset])
            parts.extend(
                item["text"] if item["type"] == "text" else item
                for item in message["content"]
            )
            text = text[offset:]
        parts.append(text)
    parts.append(f"\n</{namespace}dialog>")

    cm_content = []
    text = ""
    for part in parts:
        if isinstance(part, str):
            text += part
        else:
            if text:
                cm_content.append(dict(type="text", text=text))
                text = ""
            cm_content.append(deepcopy(part))
    if text:
        cm_content.append(dict(type="text", text=text))
    return cm_content


def apply_correct_once(policy_traj, arguments):
    """Fork a marked dialog at the end of the selected left anchor in CM text space."""
    if arguments.get("is_good") is not False:
        raise ValueError("apply_correct_once expects is_good=false")

    def not_found(feedback, match_num=0):
        return dict(
            messages_location=dict(
                not_found=True, match_num=match_num, find_feedback=feedback
            )
        )

    left = arguments.get("location_left", "")
    right = arguments.get("location_right")
    location_index = arguments.get("location_index", 0)
    replacement = arguments.get("replacement_text")
    if not isinstance(left, str) or not isinstance(right, str) or not right:
        return not_found("Expected string anchors and a nonempty location_right")
    if type(location_index) is not int:
        return not_found("location_index must be an integer")
    if not isinstance(replacement, str):
        return not_found("replacement_text must be a string")

    messages = policy_traj["messages"]
    anchor = left + right
    matches = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not message.get("sidecar", {}).get(
            "correction", {}
        ).get("_correction_area"):
            continue
        body = _render_body(message, index)
        text = body["templated_prompt"]
        pos = text.find(anchor)
        while pos != -1:
            matches.append((body, pos + len(left)))
            pos = text.find(anchor, pos + 1)
    if not matches:
        return not_found("Anchors not found in a correctable assistant message")
    if not -len(matches) <= location_index < len(matches):
        return not_found(f"location_index out of range: {location_index}", len(matches))

    body, fork = matches[location_index]
    index = body["message_index"]
    messages_location = dict(
        build_messages_location(body, fork),
        match_num=len(matches),
        find_feedback="matched",
    )
    history = deepcopy(messages[:index])
    partial_message = response_template.parse(
        body["templated_prompt"][:fork] + replacement,
        messages=history,
        tools=policy_traj.get("tools"),
    )
    return dict(
        messages_location=messages_location,
        partial_messages=history + [dict(role="assistant") | partial_message],
    )


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
    with AgentSession(ChatAPI.free_api(stream=False), cm_traj) as agent:
        loop_res = agent.run_agentic_loop()
    tree(loop_res)
    cm_response = loop_res["trajectory"]["messages"][-1]
    print(cm_response)
