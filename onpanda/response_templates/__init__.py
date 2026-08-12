#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response templates: `apply` renders an assistant message into the model's own response
text, `parse` turns that text back into a structured message. Ported from onPanda vue
`src/utils/responseTemplates`, targeted at iterative correction instead of view tokens.
"""

from bisect import bisect_left

import mximport

with mximport.inpkg():
    from .default import (
        DefaultResponseTemplate,
        TOOL_RESPONSE_MARKER,
        content_to_text,
        test_default_response_template,
    )
    from .qwen3p5 import Qwen3p5ResponseTemplate, test_qwen3p5_response_template
    from .step3p5 import Step3p5ResponseTemplate, test_step3p5_response_template

RESPONSE_TEMPLATE_CLASSES = [Qwen3p5ResponseTemplate, Step3p5ResponseTemplate]
FLATTENED_MESSAGE_KEYS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
)


def build_response_template(response_template=None, special_tokens=None):
    """
    `response_template` is the same config dict as onPanda vue, matched by `name_or_path`,
    or an already built template. Fall back to DefaultResponseTemplate when nothing matches.
    """
    if response_template is not None and not isinstance(response_template, dict):
        return response_template
    for response_template_class in RESPONSE_TEMPLATE_CLASSES:
        if response_template_class.match(response_template):
            return response_template_class(
                response_template, special_tokens=special_tokens
            )
    return DefaultResponseTemplate(response_template, special_tokens=special_tokens)


def _get_channel_text(templated_location, key_path):
    channel = templated_location["message"]
    for key in key_path:
        channel = channel[key]
    return content_to_text(channel)


def build_messages_location(templated_location, templated_char_index):
    """
    Convert a template space position into a structured messages_location, the only anchor
    independent of both tokenizer and response template, so reward and old data keep comparable.
    Scaffolding after a mapped fragment snaps to that fragment's end, while leading
    scaffolding snaps to the first fragment's start.  When mappings share a template
    position, the later mapping owns that boundary.

    `templated_location` is an `apply` result plus its `message_index` and source `message`.
    Mapping channel ranges are half-open offsets in the source value selected by `key_path`.
    `channel_boundaries`, when present, contains one absolute channel index per rendered
    text boundary and is monotonically nondecreasing.  A non-bijective normalized fragment
    without exact boundaries snaps its interior to the source fragment's start.
    """
    key_path = ["content"]
    char_index = 0
    selected_mapping = None
    mappings = templated_location["key_path_prompt_mapping"]
    for mapping in mappings:
        if templated_char_index < mapping["text_start"]:
            break
        selected_mapping = mapping
        key_path = mapping["key_path"]
    if selected_mapping is None and mappings:
        selected_mapping = mappings[0]
        key_path = selected_mapping["key_path"]
        char_index = selected_mapping["channel_start"]
    elif selected_mapping is not None:
        channel_start = selected_mapping["channel_start"]
        channel_text = _get_channel_text(templated_location, key_path)
        channel_end = selected_mapping["channel_end"]
        if templated_char_index >= selected_mapping["text_end"]:
            char_index = channel_end
        else:
            rendered_text = templated_location["templated_prompt"][
                selected_mapping["text_start"] : selected_mapping["text_end"]
            ]
            source_text = channel_text[channel_start:channel_end]
            rendered_offset = templated_char_index - selected_mapping["text_start"]
            if "channel_boundaries" in selected_mapping:
                char_index = selected_mapping["channel_boundaries"][rendered_offset]
            elif rendered_text == source_text:
                char_index = channel_start + rendered_offset
            else:
                char_index = channel_start
    channel_text = _get_channel_text(templated_location, key_path)
    return dict(
        path_keys=[templated_location["message_index"]] + list(key_path),
        char_index=char_index,
        left5=channel_text[max(0, char_index - 5) : char_index],
        right5=channel_text[char_index : char_index + 5],
    )


def build_templated_char_index(templated_location, messages_location):
    """Project a structured location into template space using the mapping's snap rules."""
    key_path = list(messages_location["path_keys"][1:])
    mappings = [
        mapping
        for mapping in templated_location["key_path_prompt_mapping"]
        if mapping["key_path"] == key_path
    ]
    assert mappings, (
        f"messages_location {messages_location} has no channel in templated prompt: "
        f"{templated_location['key_path_prompt_mapping']}"
    )
    mapping = mappings[0]
    for candidate in mappings:
        if messages_location["char_index"] < candidate["channel_start"]:
            break
        mapping = candidate
    channel_start = mapping["channel_start"]
    channel_text = _get_channel_text(templated_location, key_path)
    channel_end = mapping["channel_end"]
    source_text = channel_text[channel_start:channel_end]
    rendered_text = templated_location["templated_prompt"][
        mapping["text_start"] : mapping["text_end"]
    ]
    if messages_location["char_index"] <= channel_start:
        return mapping["text_start"]
    if messages_location["char_index"] >= channel_end:
        return mapping["text_end"]
    if "channel_boundaries" in mapping:
        boundary_index = bisect_left(
            mapping["channel_boundaries"], messages_location["char_index"]
        )
        if (
            mapping["channel_boundaries"][boundary_index]
            != messages_location["char_index"]
        ):
            # A cut inside one JSON escape belongs before its decoded character.
            boundary_index -= 1
        return mapping["text_start"] + boundary_index
    if source_text == rendered_text:
        return mapping["text_start"] + messages_location["char_index"] - channel_start
    return mapping["text_start"]


def flatten_messages_for_correcting(messages, response_template):
    """
    Flatten channels that the correcting model's own chat template would drop or restructure,
    so it sees reasoning and tool calls as plain text. A content only assistant message needs
    no flattening and stays untouched. Consecutive tool responses merge into one user message,
    because flattened tool calls leave their `tool_call_id` without a structured owner.
    """
    flattened = []
    previous_is_tool = False
    for message in messages:
        has_reasoning = message.get("reasoning") is not None or bool(
            message.get("reasoning_content")
        )
        if message["role"] == "assistant" and (
            has_reasoning or message.get("tool_calls") is not None
        ):
            flattened_message = {
                key: value
                for key, value in message.items()
                if key not in FLATTENED_MESSAGE_KEYS
            }
            template_message = message
            if message.get("reasoning_content"):
                template_message = dict(message, reasoning=message["reasoning_content"])
            flattened_message["content"] = response_template.apply(template_message)[
                "templated_prompt"
            ]
            flattened.append(flattened_message)
        elif message["role"] == "tool":
            tool_response = (
                TOOL_RESPONSE_MARKER
                + (message.get("tool_call_id") or "")
                + "\n"
                + content_to_text(message.get("content"))
            )
            if previous_is_tool:
                flattened[-1]["content"] += "\n" + tool_response
            else:
                flattened.append(dict(role="user", content=tool_response))
        else:
            flattened.append(message)
        previous_is_tool = message["role"] == "tool"
    return flattened


def test_response_templates():
    return dict(
        default=test_default_response_template(),
        qwen3p5=test_qwen3p5_response_template(),
        step3p5=test_step3p5_response_template(),
    )


if __name__ == "__main__":
    print("test_response_templates passed:", test_response_templates())
