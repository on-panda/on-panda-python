#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response templates: `apply` renders an assistant message into the model's own response
text, `parse` turns that text back into a structured message. Ported from onPanda vue
`src/utils/responseTemplates`, targeted at iterative correction instead of view tokens.
"""

import mximport

with mximport.inpkg():
    from .default import (
        DefaultResponseTemplate,
        TOOL_RESPONSE_MARKER,
        content_to_text,
        test_default_response_template,
    )
    from .qwen3p5 import Qwen3p5ResponseTemplate, test_qwen3p5_response_template

RESPONSE_TEMPLATE_CLASSES = [Qwen3p5ResponseTemplate]
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
        if message["role"] == "assistant" and (
            message.get("reasoning") is not None
            or message.get("tool_calls") is not None
        ):
            flattened_message = {
                key: value
                for key, value in message.items()
                if key not in FLATTENED_MESSAGE_KEYS
            }
            flattened_message["content"] = response_template.apply(message)[
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
    )


if __name__ == "__main__":
    print("test_response_templates passed:", test_response_templates())
