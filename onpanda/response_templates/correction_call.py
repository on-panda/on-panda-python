#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correction-call response body; message headers and trajectory formatting live outside it."""

import mximport

with mximport.inpkg():
    from .deepseek_v4 import DeepSeekV4ResponseTemplate


class CorrectionCallResponseTemplate(DeepSeekV4ResponseTemplate):
    default_name_or_path = "onpanda/correction-call"
    namespace = "｜CM｜"
    tool_call_tag_name = "function"
    parameter_tag_name = "item"
    parameter_key_attribute = "key"
    string_attribute = "str"
    string_values = ("0", "1")
    reasoning_begin_marker = "".join(["<", namespace, "reasoning>"])
    reasoning_end_marker = "".join(["</", namespace, "reasoning>"])
    content_begin_marker = "".join(["<", namespace, "content>"])
    content_end_marker = "".join(["</", namespace, "content>"])
    reasoning_content_separator = "\n"
    content_tool_calls_separator = "\n"
    eos = "".join(["</", namespace, "message>"])
    emit_eos = True

    @staticmethod
    def match(response_template=None):
        return (
            (response_template or {}).get("name_or_path") or ""
        ).lower() == "onpanda/correction-call"

    def parse(self, text, messages=None, tools=None, finish_reason=None):
        # A recognizable but unfinished protocol marker is not channel text.
        marker_start = max(
            text.rfind("<" + self.namespace), text.rfind("</" + self.namespace)
        )
        if marker_start != -1:
            suffix = text[marker_start:]
            if any(
                marker.startswith(suffix) and marker != suffix
                for marker in (
                    self.reasoning_begin_marker,
                    self.reasoning_end_marker,
                    self.content_begin_marker,
                    self.content_end_marker,
                    self.tool_calls_begin,
                    self.tool_calls_end,
                    self.invoke_prefix,
                    self.invoke_end,
                    self.parameter_prefix,
                    self.parameter_end,
                    self.eos,
                )
            ):
                text = text[:marker_start]
        return super().parse(
            text, messages=messages, tools=tools, finish_reason=finish_reason
        )


if __name__ == "__main__":
    import json
    from copy import deepcopy

    with mximport.inpkg():
        from . import (
            build_response_template,
            build_messages_location,
            build_templated_char_index,
        )
        from .consistency import template_test_trajectory

    template = build_response_template(dict(name_or_path="onpanda/correction-call"))
    assert type(template).__name__ == "CorrectionCallResponseTemplate"
    assert not template.match(dict(name_or_path="onpanda/CM"))
    assert template.apply({})["templated_prompt"] == ""
    assert template.parse("") == {}

    cm = "｜CM｜"
    message = dict(
        role="assistant",
        reasoning="think\n\n",
        content="answer <>&\n\n",
        tool_calls=[
            dict(
                function=dict(
                    name="read_file",
                    arguments='{"path": "line1\\nline2\\n", "limit": 10}',
                )
            )
        ],
        finish_reason="tool_calls",
    )
    expected = (
        f"<{cm}reasoning>think\n\n</{cm}reasoning>\n"
        f"<{cm}content>answer <>&\n\n</{cm}content>\n"
        f'<{cm}tool_calls>\n<{cm}function name="read_file">\n'
        f'<{cm}item key="path" str="1">line1\nline2\n</{cm}item>\n'
        f'<{cm}item key="limit" str="0">10</{cm}item>\n'
        f"</{cm}function>\n</{cm}tool_calls>\n</{cm}message>"
    )
    assert template.apply(message)["templated_prompt"] == expected
    parsed = template.parse(expected)
    assert parsed["reasoning"] == message["reasoning"]
    assert parsed["content"] == message["content"]
    assert parsed["finish_reason"] == "tool_calls"
    assert parsed["tool_calls"][0]["function"] == message["tool_calls"][0]["function"]
    assert template.apply(parsed)["templated_prompt"] == expected
    assert template.parse(expected, finish_reason="length")["finish_reason"] == "length"

    # Channel presence matters even when the channel has no text yet.
    for message in (
        dict(role="assistant"),
        dict(role="assistant", reasoning=""),
        dict(role="assistant", content=""),
        dict(role="assistant", reasoning="think\n", content=""),
        dict(role="assistant", reasoning="think\n", finish_reason="reasoning_end"),
        dict(role="assistant", content="answer\n"),
        dict(role="assistant", content="", finish_reason="stop"),
        dict(role="assistant", tool_calls=[{}]),
        dict(role="assistant", tool_calls=[dict(function=dict(name="read_fi"))]),
    ):
        text = template.apply(message)["templated_prompt"]
        parsed = template.parse(text)
        assert template.apply(parsed)["templated_prompt"] == text, (message, parsed)
        if "content" in message:
            assert parsed["content"] == message["content"]
        else:
            assert "content" not in parsed
    content_text = (
        template.content_begin_marker + "answer\n" + template.content_end_marker
    )
    assert "finish_reason" not in template.parse(content_text)
    assert template.parse(content_text + "\n" + template.eos)["finish_reason"] == "stop"
    for channel, begin, end in (
        ("reasoning", template.reasoning_begin_marker, template.reasoning_end_marker),
        ("content", template.content_begin_marker, template.content_end_marker),
    ):
        parsed = template.parse(begin + "body\n" + template.eos)
        assert parsed[channel] == "body\n" and parsed["finish_reason"] == "stop"
        assert template.parse(begin + "body\n" + end[:-1])[channel] == "body\n"
        for literal in ("<", "</", "</｜C", "</｜CM｜unknown"):
            assert (
                template.parse(begin + "body\n" + literal)[channel]
                == "body\n" + literal
            )

    # Partial tags must not turn already parsed tool arguments back into raw markup.
    tool_prefix = (
        template.tool_calls_begin
        + "\n"
        + template.invoke_prefix
        + 'read_file">\n'
        + template.parameter_prefix
        + 'path" str="1">abc'
    )
    for suffix in (
        template.parameter_end[:-1],
        template.parameter_end + "\n" + template.invoke_end[:-1],
        template.parameter_end + "\n" + template.parameter_prefix[:-1],
    ):
        arguments = template.parse(tool_prefix + suffix)["tool_calls"][0]["function"][
            "arguments"
        ]
        assert arguments in ('{"path": "abc', '{"path": "abc", '), arguments

    # No schema is needed to recover JSON types from str="0".
    message = deepcopy(template_test_trajectory["messages"][4])
    text = template.apply(message)["templated_prompt"]
    parsed = template.parse(text)
    for original, recovered in zip(message["tool_calls"], parsed["tool_calls"]):
        assert json.loads(original["function"]["arguments"]) == json.loads(
            recovered["function"]["arguments"]
        )
    assert template.apply(parsed)["templated_prompt"] == text
    for arguments in (
        '{"pa',
        '{"path"',
        '{"path": "line1\\n',
        '{"limit": 1',
        '{"flag": true',
        '{"obj": {"a": 1}',
        "{}",
    ):
        message = dict(
            role="assistant",
            tool_calls=[dict(function=dict(name="read_file", arguments=arguments))],
        )
        text = template.apply(message)["templated_prompt"]
        assert template.apply(template.parse(text))["templated_prompt"] == text

    message = dict(
        role="assistant",
        reasoning="think\n",
        content="answer\n",
        tool_calls=[
            dict(function=dict(name="write", arguments='{"text": "a\\n\\u4e2d"}'))
        ],
        finish_reason="tool_calls",
    )
    applied = dict(template.apply(message), message_index=0, message=message)
    for mapping in applied["key_path_prompt_mapping"]:
        for offset in range(mapping["text_end"] - mapping["text_start"]):
            index = mapping["text_start"] + offset
            location = build_messages_location(applied, index)
            expected_index = (
                mapping["channel_boundaries"][offset]
                if "channel_boundaries" in mapping
                else mapping["channel_start"] + offset
            )
            assert location["path_keys"] == [0] + mapping["key_path"]
            assert location["char_index"] == expected_index
            assert build_templated_char_index(applied, location) == index
    print(
        "CorrectionCallResponseTemplate passed: protocol, finish reasons, whitespace, partials, and mappings"
    )
