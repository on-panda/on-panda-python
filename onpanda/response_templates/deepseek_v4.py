#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek V4 response text and DSML tool calls, matching the onPanda annotator."""

import json
import re

import mximport

with mximport.inpkg():
    from .partial_json import parse_partial_json_object
    from .qwen3p5 import (
        COMPLETE_FINISH_REASONS,
        REASONING_END,
        THINK_BEGIN,
        THINK_END,
        normalize_message_tool_calls,
    )


class DeepSeekV4ResponseTemplate:
    default_name_or_path = "deepseek-ai/DeepSeek-V4-Flash"
    namespace = "｜DSML｜"
    tool_calls_block_name = "tool_calls"
    tool_call_tag_name = "invoke"
    parameter_tag_name = "parameter"
    parameter_key_attribute = "name"
    string_attribute = "string"
    string_values = ("false", "true")
    reasoning_begin_marker = THINK_BEGIN
    reasoning_end_marker = THINK_END
    content_begin_marker = ""
    content_end_marker = ""
    reasoning_content_separator = ""
    content_tool_calls_separator = "\n\n"
    eos = "".join(["<｜", "end▁of▁sentence", "｜>"])
    emit_eos = False

    @staticmethod
    def match(response_template=None):
        return bool(
            re.match(
                r"(?:deepseek-ai/)?DeepSeek-V4(?:-(?:Flash|Pro)(?:-|$)|$)",
                (response_template or {}).get("name_or_path") or "",
                re.IGNORECASE,
            )
        )

    def __init__(self, response_template=None, special_tokens=None):
        self.config = response_template or {}
        self.name_or_path = self.config.get("name_or_path") or self.default_name_or_path
        self.tool_calls_begin = "".join(
            ["<", self.namespace, self.tool_calls_block_name, ">"]
        )
        self.tool_calls_end = "".join(
            ["</", self.namespace, self.tool_calls_block_name, ">"]
        )
        self.invoke_prefix = "".join(
            ["<", self.namespace, self.tool_call_tag_name, ' name="']
        )
        self.invoke_end = "".join(["</", self.namespace, self.tool_call_tag_name, ">"])
        self.parameter_prefix = "".join(
            [
                "<",
                self.namespace,
                self.parameter_tag_name,
                f' {self.parameter_key_attribute}="',
            ]
        )
        self.parameter_end = "".join(
            ["</", self.namespace, self.parameter_tag_name, ">"]
        )

    def __str__(self):
        return f"{type(self).__name__}({self.config})"

    __repr__ = __str__

    def apply(self, message):
        templated_prompt = ""
        key_path_prompt_mapping = []

        def append_mapped(
            key_path, text, channel_start=0, channel_end=None, channel_boundaries=None
        ):
            nonlocal templated_prompt
            mapping = dict(
                key_path=key_path,
                text_start=len(templated_prompt),
                text_end=len(templated_prompt) + len(text),
                channel_start=channel_start,
                channel_end=(
                    channel_start + len(text) if channel_end is None else channel_end
                ),
            )
            if channel_boundaries is not None:
                mapping["channel_boundaries"] = channel_boundaries
            key_path_prompt_mapping.append(mapping)
            templated_prompt += text

        is_partial = message.get("finish_reason") not in COMPLETE_FINISH_REASONS
        reasoning = message.get("reasoning") or ""
        while reasoning.startswith(self.reasoning_begin_marker):
            reasoning = reasoning[len(self.reasoning_begin_marker) :]
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        if not message.get("role") and not (reasoning or content or tool_calls):
            return dict(templated_prompt="", key_path_prompt_mapping=[])
        is_pure_reasoning_partial = (
            is_partial
            and message.get("finish_reason") != REASONING_END
            and not content
            and not tool_calls
            and not (self.content_begin_marker and "content" in message)
        )
        has_reasoning = (
            reasoning
            or is_pure_reasoning_partial
            or (self.content_begin_marker and "reasoning" in message)
        )
        if has_reasoning:
            templated_prompt += self.reasoning_begin_marker
        if "reasoning" in message:
            append_mapped(
                ["reasoning"],
                reasoning,
                len(message["reasoning"] or "") - len(reasoning),
            )
        if not is_pure_reasoning_partial and (
            has_reasoning or not self.content_begin_marker
        ):
            templated_prompt += self.reasoning_end_marker
        if "content" in message:
            if has_reasoning:
                templated_prompt += self.reasoning_content_separator
            templated_prompt += self.content_begin_marker
            append_mapped(["content"], content)
            if not is_partial or tool_calls:
                templated_prompt += self.content_end_marker

        if tool_calls:
            if templated_prompt:
                templated_prompt += self.content_tool_calls_separator
            templated_prompt += self.tool_calls_begin
            if tool_calls != [{}]:
                templated_prompt += "\n"
            for position, tool_call in enumerate(
                [] if tool_calls == [{}] else tool_calls
            ):
                function = tool_call.get("function") or {}
                templated_prompt += self.invoke_prefix
                if "name" in function:
                    append_mapped(
                        ["tool_calls", position, "function", "name"], function["name"]
                    )
                if "arguments" not in function:
                    continue
                templated_prompt += '">\n'
                arguments = function["arguments"]
                arguments_path = ["tool_calls", position, "function", "arguments"]
                append_mapped(arguments_path, "", 0, 0)
                parsed_arguments = parse_partial_json_object(arguments)
                if parsed_arguments:
                    for parameter_index, parameter in enumerate(
                        parsed_arguments["entries"]
                    ):
                        if parameter_index:
                            templated_prompt += "\n"
                        templated_prompt += self.parameter_prefix
                        append_mapped(
                            arguments_path,
                            parameter["name"],
                            parameter["name_start"],
                            parameter["name_end"],
                            parameter["name_offsets"],
                        )
                        if not parameter["name_complete"]:
                            break
                        templated_prompt += '"'
                        if "value" not in parameter:
                            break
                        # String offsets include decoded escape boundaries; nonstrings stay JSON.
                        is_string = "value_offsets" in parameter
                        templated_prompt += f' {self.string_attribute}="{self.string_values[is_string]}">'
                        value = parameter["value"]
                        if parameter["complete"] and not is_string:
                            value = json.dumps(value, ensure_ascii=False)
                        append_mapped(
                            arguments_path,
                            value,
                            parameter["value_start"],
                            parameter["value_end"],
                            parameter.get("value_offsets"),
                        )
                        if not parameter["complete"]:
                            break
                        templated_prompt += self.parameter_end
                else:
                    append_mapped(arguments_path, arguments)
                append_mapped(arguments_path, "", len(arguments), len(arguments))
                is_last_partial_tool_call = (
                    is_partial and position == len(tool_calls) - 1
                )
                if not is_last_partial_tool_call or (
                    parsed_arguments and parsed_arguments["complete"]
                ):
                    if not parsed_arguments and arguments:
                        templated_prompt += "\n"
                    templated_prompt += "\n" + self.invoke_end
                if position < len(tool_calls) - 1:
                    templated_prompt += "\n"
            if not is_partial and tool_calls != [{}]:
                templated_prompt += "\n" + self.tool_calls_end
        if not is_partial and self.emit_eos:
            templated_prompt += ("\n" if templated_prompt else "") + self.eos
        return dict(
            templated_prompt=templated_prompt,
            key_path_prompt_mapping=key_path_prompt_mapping,
        )

    def parse(self, text, messages=None, tools=None, finish_reason=None):
        if not text:
            return {}
        if text.endswith(self.eos + "\n"):
            text = text[:-1]
        has_eos = text.endswith(self.eos)
        if has_eos:
            text = text[: -len(self.eos)]
            if self.emit_eos:
                finish_reason = finish_reason or "stop"
        message = dict(role="assistant")
        reasoning_closed = False
        if text.startswith(self.reasoning_begin_marker):
            text = text[len(self.reasoning_begin_marker) :]
            while text.startswith(self.reasoning_begin_marker):
                text = text[len(self.reasoning_begin_marker) :]
            reasoning_end = text.find(self.reasoning_end_marker)
            tool_calls_start = (
                -1 if self.content_begin_marker else text.find(self.tool_calls_begin)
            )
            if reasoning_end == -1 and tool_calls_start == -1:
                # Even an empty prefix is in reasoning, which the continuation caller must know.
                message["reasoning"] = text
                if finish_reason:
                    message["finish_reason"] = finish_reason
                return message
            end = tool_calls_start if reasoning_end == -1 else reasoning_end
            reasoning = text[:end]
            if not self.content_begin_marker:
                reasoning = reasoning.rstrip("\n")
            if reasoning or self.content_begin_marker:
                message["reasoning"] = reasoning
            text = text[
                end + (0 if reasoning_end == -1 else len(self.reasoning_end_marker)) :
            ]
            text = text.removeprefix(self.reasoning_content_separator)
            reasoning_closed = True
        elif not self.content_begin_marker and self.reasoning_end_marker in text:
            reasoning_end = text.index(self.reasoning_end_marker)
            if reasoning_end:
                message["reasoning"] = text[:reasoning_end]
            text = text[reasoning_end + len(self.reasoning_end_marker) :]
            reasoning_closed = True

        if self.content_begin_marker and text.startswith(self.content_begin_marker):
            content_start = len(self.content_begin_marker)
            content_end = text.find(self.content_end_marker, content_start)
            if content_end == -1:
                message["content"] = text[content_start:]
                text = ""
            else:
                message["content"] = text[content_start:content_end]
                text = text[content_end + len(self.content_end_marker) :].removeprefix(
                    self.content_tool_calls_separator
                )
        tool_calls_start = text.find(self.tool_calls_begin)
        if not self.content_begin_marker:
            message["content"] = (
                text if tool_calls_start == -1 else text[:tool_calls_start].rstrip("\n")
            )
        if tool_calls_start != -1:
            text = text[tool_calls_start + len(self.tool_calls_begin) :]
            section_end = text.find(self.tool_calls_end)
            if section_end != -1:
                text = text[:section_end]
            tool_calls = []
            cursor = 0
            while cursor < len(text):
                invoke_start = text.find(self.invoke_prefix, cursor)
                if invoke_start == -1:
                    break
                name_start = invoke_start + len(self.invoke_prefix)
                name_end = text.find('"', name_start)
                name = text[name_start : len(text) if name_end == -1 else name_end]
                tool_call = dict(
                    type="function", index=len(tool_calls), function=dict(name=name)
                )
                tool_calls.append(tool_call)
                if name_end == -1 or text[name_end + 1 : name_end + 2] != ">":
                    break
                body_start = name_end + 2
                next_invoke = text.find(self.invoke_prefix, body_start)
                invoke_close = text.find(self.invoke_end, body_start)
                body_end = min(
                    index
                    for index in (next_invoke, invoke_close, len(text))
                    if index != -1
                )
                function_closed = invoke_close != -1 and invoke_close == body_end
                raw_arguments = text[body_start:body_end].removeprefix("\n")
                if function_closed:
                    raw_arguments = raw_arguments.removesuffix("\n")
                argument_parts = []
                parameter_cursor = 0
                parameter_complete = False
                while parameter_cursor < len(raw_arguments):
                    parameter_start = raw_arguments.find(
                        self.parameter_prefix, parameter_cursor
                    )
                    if parameter_start == -1:
                        if raw_arguments[parameter_cursor:].strip():
                            argument_parts = None
                        break
                    if raw_arguments[parameter_cursor:parameter_start].strip():
                        argument_parts = None
                        break
                    key_start = parameter_start + len(self.parameter_prefix)
                    key_end = raw_arguments.find('"', key_start)
                    parameter_complete = False
                    if key_end == -1:
                        argument_parts.append(
                            json.dumps(raw_arguments[key_start:], ensure_ascii=False)[
                                :-1
                            ]
                        )
                        break
                    key_text = json.dumps(
                        raw_arguments[key_start:key_end], ensure_ascii=False
                    )
                    marker = re.match(
                        rf'"\s+{self.string_attribute}="({"|".join(self.string_values)})">',
                        raw_arguments[key_end:],
                    )
                    if not marker:
                        argument_parts.append(key_text)
                        break
                    value_start = key_end + marker.end()
                    next_parameter = raw_arguments.find(
                        self.parameter_prefix, value_start
                    )
                    parameter_close = raw_arguments.find(
                        self.parameter_end, value_start
                    )
                    value_end = min(
                        index
                        for index in (
                            next_parameter,
                            parameter_close,
                            len(raw_arguments),
                        )
                        if index != -1
                    )
                    has_parameter_close = (
                        parameter_close != -1 and parameter_close == value_end
                    )
                    parameter_complete = (
                        has_parameter_close or next_parameter != -1 or function_closed
                    )
                    value = raw_arguments[value_start:value_end]
                    if marker[1] == self.string_values[True]:
                        value = json.dumps(value, ensure_ascii=False)
                        if not parameter_complete:
                            value = value[:-1]
                    argument_parts.append(f"{key_text}: {value}")
                    parameter_cursor = value_end + (
                        len(self.parameter_end) if has_parameter_close else 0
                    )
                if argument_parts is None:
                    arguments = raw_arguments
                elif not argument_parts:
                    arguments = "{}" if function_closed else ""
                else:
                    arguments = "{" + ", ".join(argument_parts)
                    if function_closed and parameter_complete:
                        arguments += "}"
                    elif parameter_complete:
                        arguments += ", "
                tool_call["function"]["arguments"] = arguments
                if next_invoke != -1 and next_invoke == body_end:
                    cursor = next_invoke
                elif function_closed:
                    cursor = invoke_close + len(self.invoke_end)
                else:
                    break
            message["tool_calls"] = tool_calls or [{}]
        if has_eos:
            message["finish_reason"] = "stop"
        elif (
            reasoning_closed
            and not message.get("content")
            and not (self.content_begin_marker and "content" in message)
            and "tool_calls" not in message
        ):
            message["finish_reason"] = REASONING_END
        if finish_reason:
            message["finish_reason"] = finish_reason
        if message.get("finish_reason") == "stop" and any(
            call.get("function", {}).get("name")
            for call in message.get("tool_calls", [])
        ):
            message["finish_reason"] = "tool_calls"
        return normalize_message_tool_calls(message, messages)


if __name__ == "__main__":
    from types import SimpleNamespace
    from unittest.mock import Mock

    with mximport.inpkg():
        from .deepseek_v4p1 import DeepSeekV4p1ResponseTemplate
        from . import (
            build_response_template,
            build_messages_location,
            build_templated_char_index,
        )
        from .consistency import template_test_trajectory
        from .. import FindAndReplaceCorrectionAdapter, utf8_tokenizer
        from ..correcting_model.correcting_model import CorrectingModel
        from ..test_utils import get_test_trajectories

    for template_class in (DeepSeekV4ResponseTemplate, DeepSeekV4p1ResponseTemplate):
        template = template_class()
        assert (
            type(
                build_response_template(dict(name_or_path=template.name_or_path))
            ).__name__
            == template_class.__name__
        )
        assert template.parse("") == {}
        assert template.apply({})["templated_prompt"] == ""
        assert template.apply(dict(role="assistant"))["templated_prompt"] == THINK_BEGIN
        assert template.parse(THINK_BEGIN + "thinking\n") == dict(
            role="assistant", reasoning="thinking\n"
        )
        assert (
            template.parse(THINK_BEGIN + "thinking" + THINK_END)["finish_reason"]
            == REASONING_END
        )
        assert (
            template.apply(dict(role="assistant", content="answer"))["templated_prompt"]
            == THINK_END + "answer"
        )
        block, invoke, parameter = (
            ("tool_calls", "invoke", "parameter")
            if template_class is DeepSeekV4ResponseTemplate
            else (" calls", " invoke", " parameter")
        )
        dsml = "｜DSML｜"
        expected = (
            f'{THINK_END}answer\n\n<{dsml}{block}>\n<{dsml}{invoke} name="read">\n'
            f'<{dsml}{parameter} name="path" string="true">/tmp</{dsml}{parameter}>\n'
            f'<{dsml}{parameter} name="limit" string="false">10</{dsml}{parameter}>\n'
            f"</{dsml}{invoke}>\n</{dsml}{block}>"
        )
        assert (
            template.apply(
                dict(
                    role="assistant",
                    content="answer",
                    finish_reason="tool_calls",
                    tool_calls=[
                        dict(
                            function=dict(
                                name="read", arguments='{"path": "/tmp", "limit": 10}'
                            )
                        )
                    ],
                )
            )["templated_prompt"]
            == expected
        )
        assert template.parse(expected, finish_reason="stop")["tool_calls"][0][
            "function"
        ] == dict(name="read", arguments='{"path": "/tmp", "limit": 10}')

        message = template_test_trajectory["messages"][4]
        applied = template.apply(message)
        parsed = template.parse(
            applied["templated_prompt"] + template.eos,
            messages=template_test_trajectory["messages"][:4],
        )
        assert parsed["finish_reason"] == "tool_calls"
        for original_call, parsed_call in zip(
            message["tool_calls"], parsed["tool_calls"]
        ):
            assert original_call["function"]["name"] == parsed_call["function"]["name"]
            assert json.loads(original_call["function"]["arguments"]) == json.loads(
                parsed_call["function"]["arguments"]
            )
        assert template.apply(parsed)["templated_prompt"] == applied["templated_prompt"]

        for arguments in (
            "",
            "{",
            '{"',
            '{"pa',
            '{"path"',
            '{"path":',
            '{"path": "',
            '{"path": "/tm',
            '{"path": "/tmp"',
            '{"path": "/tmp",',
            '{"limit": 10',
            '{"flag": tr',
            '{"flag": true',
            '{"list": [1, 2',
            '{"obj": {"a": 1}',
            '{"text": "say \\"hi',
            '{"text": "line1\\n',
            '{"value": null}',
            "{}",
        ):
            partial = dict(
                role="assistant",
                tool_calls=[dict(function=dict(name="read_file", arguments=arguments))],
            )
            text = template.apply(partial)["templated_prompt"]
            parsed = template.parse(text, finish_reason="length")
            assert parsed["finish_reason"] == "length"
            assert template.apply(parsed)["templated_prompt"] == text, (
                arguments,
                parsed,
            )
        # A complete nonstring value can close canonically; a trailing number can still grow.
        for value in ("true", "false", "null", "[1, 2]", '{"a": 1}', "10"):
            prefix = (
                THINK_END
                + "\n\n"
                + template.tool_calls_begin
                + "\n"
                + template.invoke_prefix
                + 'read">\n'
                + template.parameter_prefix
                + 'value" string="false">'
                + value
            )
            assert template.apply(template.parse(prefix))["templated_prompt"] == (
                prefix if value == "10" else prefix + template.parameter_end
            )

        for partial in (
            dict(role="assistant", tool_calls=[{}]),
            dict(role="assistant", tool_calls=[dict(function=dict(name="read_fi"))]),
        ):
            text = template.apply(partial)["templated_prompt"]
            assert template.apply(template.parse(text))["templated_prompt"] == text

        arguments = '{"text": "line1\\nquote\\"中文\\uD83D\\uDE00"}'
        message = dict(
            role="assistant",
            tool_calls=[dict(function=dict(name="write", arguments=arguments))],
        )
        applied = dict(template.apply(message), message_index=0, message=message)
        for mapping in applied["key_path_prompt_mapping"]:
            if "channel_boundaries" not in mapping:
                continue
            for offset, channel_index in enumerate(mapping["channel_boundaries"]):
                text_index = mapping["text_start"] + offset
                location = build_messages_location(applied, text_index)
                if offset < len(mapping["channel_boundaries"]) - 1:
                    assert location["char_index"] == channel_index
                    assert build_templated_char_index(applied, location) == text_index
        for trajectory in get_test_trajectories().values():
            for correction in trajectory["corrections"]:
                fork = correction["fork_message_index"]
                if fork is None:
                    continue
                partial = correction["partial_messages"][fork]
                text = template.apply(partial)["templated_prompt"]
                parsed = template.parse(
                    text,
                    messages=trajectory["messages"][:fork],
                    finish_reason=partial.get("finish_reason"),
                )
                assert template.apply(parsed)["templated_prompt"] == text

        adapter = FindAndReplaceCorrectionAdapter(
            tokenizer=utf8_tokenizer,
            response_template=template,
            max_replacement_tokens=1000,
        )
        correcting_model = CorrectingModel(SimpleNamespace(model="correcting"), adapter)
        user_message = dict(role="user", content="q")
        rejected = dict(role="assistant", content="wrong", finish_reason="stop")
        tool_partial = dict(
            role="assistant",
            tool_calls=[dict(function=dict(name="read", arguments='{"path": "/tm'))],
        )
        for partial, reasoning_delta, content_delta, expected in (
            (
                dict(role="assistant"),
                "think",
                "answer",
                dict(reasoning="think", content="answer"),
            ),
            (
                dict(role="assistant", reasoning="start"),
                " end",
                "answer",
                dict(reasoning="start end", content="answer"),
            ),
            (
                dict(role="assistant", reasoning="think", content="start"),
                " end",
                "!",
                dict(reasoning="think", content="start end!"),
            ),
            (
                tool_partial,
                "p",
                template.parameter_end
                + "\n"
                + template.invoke_end
                + "\n"
                + template.tool_calls_end,
                dict(content="", finish_reason="tool_calls"),
            ),
        ):
            prefix = adapter.build_partial_templated_prompt(rejected, partial)[
                "templated_prompt"
            ]
            correcting_model.correct = Mock(
                return_value=dict(
                    correction=dict(
                        messages_location=dict(path_keys=[1, "content"], char_index=0)
                    ),
                    partial_messages=[user_message, partial],
                )
            )
            policy = Mock(
                return_value=dict(
                    trajectory=dict(
                        messages=[
                            dict(
                                role="assistant",
                                content=prefix + content_delta,
                                reasoning=reasoning_delta,
                                finish_reason="stop",
                            )
                        ]
                    )
                )
            )
            policy.model = template.name_or_path
            corrected = correcting_model.correct_and_rollout(
                [user_message, rejected], policy
            )["corrected_messages"][-1]
            for key, value in expected.items():
                assert corrected[key] == value, corrected
            if partial is tool_partial:
                assert corrected["tool_calls"][0]["function"] == dict(
                    name="read", arguments='{"path": "/tmp"}'
                )
            request = policy.call_args.args[0]["messages"][-1]
            assert request == dict(role="assistant", content=prefix, prefix=True)
            assert policy.call_args.kwargs["continue_final_message"] is True
            assert policy.call_args.kwargs["skip_special_tokens"] is False
        print(
            template_class.__name__,
            "passed: protocol, partials, mappings, correction trajectories, continuation",
        )

    for name in (
        "DeepSeek-V4",
        "deepseek-ai/DeepSeek-V4-Flash-FP8",
        "deepseek-ai/DeepSeek-V4-Pro-0813",
    ):
        assert DeepSeekV4ResponseTemplate.match(dict(name_or_path=name))
        assert not DeepSeekV4p1ResponseTemplate.match(dict(name_or_path=name))
    assert DeepSeekV4p1ResponseTemplate.match(
        dict(name_or_path="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    )
    assert not DeepSeekV4ResponseTemplate.match(
        dict(name_or_path="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    )
    assert not DeepSeekV4ResponseTemplate.match(
        dict(name_or_path="deepseek-ai/DeepSeek-V4-Reasoner")
    )
