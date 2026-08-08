#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response template of the Qwen3.5+ family, whose tool calls are XML parameter blocks.

Ported from onPanda vue `src/utils/responseTemplates/qwen3p5ResponseTemplate.js`,
keep both sides identical: python consumes the `.panda.json` that vue produces, so any
drift between the two implementations is a silent data error.
"""

import json
import math
import re
import mximport

with mximport.inpkg():
    from .partial_json import parse_partial_json_object


def _special_marker(name):
    # Assemble markers from fragments, so no complete marker literal appears in source.
    return "".join(["<|", name, "|>"])


def _xml_marker(name, closing=False):
    return "".join(["<", "/" if closing else "", name, ">"])


def _xml_value_marker(name):
    return "".join(["<", name, "="])


ASSISTANT_BEGIN = _special_marker("im_start") + "assistant\n"
IM_END = _special_marker("im_end")
THINK_BEGIN = _xml_marker("think")
THINK_END = _xml_marker("think", True)
TOOL_CALL_BEGIN = _xml_marker("tool_call")
TOOL_CALL_END = _xml_marker("tool_call", True)
FUNCTION_BEGIN = _xml_value_marker("function")
FUNCTION_END = _xml_marker("function", True)
PARAMETER_BEGIN = _xml_value_marker("parameter")
PARAMETER_END = _xml_marker("parameter", True)
REASONING_END = "reasoning_end"
COMPLETE_FINISH_REASONS = ("stop", "tool_calls")


def strip_repeated_think_begin(text):
    while text.startswith(THINK_BEGIN):
        text = text[len(THINK_BEGIN) :]
        if text.startswith("\n"):
            text = text[1:]
    return text


def _stringify_json_for_template(value):
    if isinstance(value, list):
        return (
            f"[{', '.join([_stringify_json_for_template(child) for child in value])}]"
        )
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                [
                    f"{json.dumps(key, ensure_ascii=False)}: {_stringify_json_for_template(child)}"
                    for key, child in value.items()
                ]
            )
            + "}"
        )
    return json.dumps(value, ensure_ascii=False)


def _argument_value_to_text(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list)):
        return _stringify_json_for_template(value)
    return str(value)


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _parameter_value_to_json_text(value, schema=None):
    string_value = json.dumps(value, ensure_ascii=False)
    if not schema:
        return string_value
    schema_type = schema.get("type")
    schema_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if len(schema_types) != 1 or not isinstance(schema_types[0], str):
        return string_value
    schema_type = schema_types[0]
    if schema_type == "string":
        return string_value
    if schema_type == "boolean":
        if value in ("True", "true"):
            return "true"
        if value in ("False", "false"):
            return "false"
        return string_value
    if schema_type == "null":
        return "null" if value in ("None", "null") else string_value

    try:
        parsed_value = json.loads(value)
    except ValueError:
        return string_value
    if schema_type == "number":
        return value if _is_finite_number(parsed_value) else string_value
    if schema_type == "integer":
        return (
            value
            if _is_finite_number(parsed_value) and float(parsed_value).is_integer()
            else string_value
        )
    if schema_type == "array":
        return value if isinstance(parsed_value, list) else string_value
    if schema_type == "object":
        return value if isinstance(parsed_value, dict) else string_value
    return string_value


def _build_arguments_prefix(parameters, function_closed, parameters_schema=None):
    properties = (parameters_schema or {}).get("properties") or {}
    argument_parts = []
    for parameter in parameters:
        if not parameter["name_complete"]:
            # The parameter name is still open, so the JSON key quote stays open too.
            argument_parts.append(
                json.dumps(parameter["name"], ensure_ascii=False)[:-1]
            )
            continue
        # An unclosed value may still grow, so its type is only known from the schema once closed.
        if parameter["complete"]:
            value_text = _parameter_value_to_json_text(
                parameter["value"], properties.get(parameter["name"])
            )
        else:
            value_text = json.dumps(parameter["value"], ensure_ascii=False)[:-1]
        argument_parts.append(
            f"{json.dumps(parameter['name'], ensure_ascii=False)}: {value_text}"
        )
    if function_closed:
        return "{" + ", ".join(argument_parts) + "}"
    # An open function keeps the separator after a closed value, otherwise a trailing
    # number would read as still growing.
    return (
        "{"
        + ", ".join(argument_parts)
        + (", " if parameters[-1].get("complete") else "")
    )


def _parse_xml_parameters(raw_arguments, function_closed, parameters_schema=None):
    parameters = []
    cursor = 0
    while cursor < len(raw_arguments):
        parameter_begin = raw_arguments.find(PARAMETER_BEGIN, cursor)
        if parameter_begin == -1:
            if raw_arguments[cursor:].strip():
                return None
            break
        if raw_arguments[cursor:parameter_begin].strip():
            return None
        name_start = parameter_begin + len(PARAMETER_BEGIN)
        name_end = raw_arguments.find(">", name_start)
        if name_end == -1:
            parameters.append(
                dict(name=raw_arguments[name_start:], name_complete=False)
            )
            break
        value_start = name_end + 1
        parameter_end = raw_arguments.find(PARAMETER_END, value_start)
        next_parameter_begin = raw_arguments.find(PARAMETER_BEGIN, value_start)
        has_parameter_end = parameter_end != -1 and (
            next_parameter_begin == -1 or parameter_end < next_parameter_begin
        )
        if has_parameter_end:
            value_end = parameter_end
        else:
            value_end = (
                len(raw_arguments)
                if next_parameter_begin == -1
                else next_parameter_begin
            )
        value = raw_arguments[value_start:value_end]
        if value.startswith("\n"):
            value = value[1:]
        parameter_complete = (
            has_parameter_end or next_parameter_begin != -1 or function_closed
        )
        if parameter_complete and value.endswith("\n"):
            value = value[:-1]
        parameters.append(
            dict(
                name=raw_arguments[name_start:name_end],
                name_complete=True,
                value=value,
                complete=parameter_complete,
            )
        )
        cursor = parameter_end + len(PARAMETER_END) if has_parameter_end else value_end

    if not parameters and raw_arguments.strip():
        return None
    if not parameters:
        return "{}" if function_closed else ""
    return _build_arguments_prefix(parameters, function_closed, parameters_schema)


def _build_tool_call(name, index, arguments_text=None):
    tool_call = dict(type="function", index=index, function=dict(name=name))
    if arguments_text is not None:
        # A missing arguments key means the arguments channel has not started, so the
        # function name is still open.
        tool_call["function"]["arguments"] = arguments_text
    return tool_call


def parse_tool_calls(tool_calls_text, tools=None):
    tools = tools or []
    tool_calls = []
    cursor = 0
    while cursor < len(tool_calls_text):
        tool_call_begin = tool_calls_text.find(TOOL_CALL_BEGIN, cursor)
        if tool_call_begin == -1:
            break
        function_begin = tool_calls_text.find(
            FUNCTION_BEGIN, tool_call_begin + len(TOOL_CALL_BEGIN)
        )
        if function_begin == -1:
            break
        name_start = function_begin + len(FUNCTION_BEGIN)
        name_end = tool_calls_text.find(">", name_start)
        if name_end == -1:
            tool_calls.append(
                _build_tool_call(tool_calls_text[name_start:], len(tool_calls))
            )
            break

        function_name = tool_calls_text[name_start:name_end]
        tool = next(
            (tool for tool in tools if tool["function"]["name"] == function_name), None
        )
        function_end = tool_calls_text.find(FUNCTION_END, name_end + 1)
        next_tool_call_begin = tool_calls_text.find(TOOL_CALL_BEGIN, name_end + 1)
        function_closed = function_end != -1 and (
            next_tool_call_begin == -1 or function_end < next_tool_call_begin
        )
        if function_closed:
            raw_arguments_end = function_end
        else:
            raw_arguments_end = (
                len(tool_calls_text)
                if next_tool_call_begin == -1
                else next_tool_call_begin
            )
        raw_arguments = tool_calls_text[name_end + 1 : raw_arguments_end]
        arguments_text = _parse_xml_parameters(
            raw_arguments,
            function_closed,
            tool["function"].get("parameters") if tool else None,
        )
        if arguments_text is None:
            arguments_text = (
                raw_arguments[1:] if raw_arguments.startswith("\n") else raw_arguments
            )
            if function_closed and arguments_text.endswith("\n"):
                arguments_text = arguments_text[:-1]
        tool_calls.append(
            _build_tool_call(function_name, len(tool_calls), arguments_text)
        )

        if next_tool_call_begin != -1 and (
            not function_closed or next_tool_call_begin < function_end
        ):
            cursor = next_tool_call_begin
            continue
        if not function_closed:
            break
        tool_call_end = tool_calls_text.find(
            TOOL_CALL_END, function_end + len(FUNCTION_END)
        )
        if tool_call_end == -1:
            break
        cursor = tool_call_end + len(TOOL_CALL_END)
    return tool_calls


def parse_qwen_response_text(text, tools=None):
    message = dict(role="assistant")
    remaining_text = text
    has_assistant_begin = False
    has_im_end = False

    if remaining_text.startswith(ASSISTANT_BEGIN):
        remaining_text = remaining_text[len(ASSISTANT_BEGIN) :]
        has_assistant_begin = True
    if remaining_text.endswith(IM_END + "\n"):
        remaining_text = remaining_text[: -len(IM_END) - 1]
        has_im_end = True
    elif remaining_text.endswith(IM_END):
        remaining_text = remaining_text[: -len(IM_END)]
        has_im_end = True

    reasoning_closed = False
    if remaining_text.startswith(THINK_BEGIN):
        reasoning_start = len(THINK_BEGIN) + (
            1 if remaining_text[len(THINK_BEGIN) : len(THINK_BEGIN) + 1] == "\n" else 0
        )
        reasoning_end = remaining_text.find(THINK_END, reasoning_start)
        if reasoning_end == -1:
            implicit_reasoning_end = remaining_text.find(
                TOOL_CALL_BEGIN, reasoning_start
            )
            reasoning = re.sub(
                r"\n+$",
                "",
                strip_repeated_think_begin(
                    remaining_text[
                        reasoning_start : (
                            len(remaining_text)
                            if implicit_reasoning_end == -1
                            else implicit_reasoning_end
                        )
                    ]
                ),
            )
            if reasoning:
                message["reasoning"] = reasoning
            if implicit_reasoning_end == -1:
                return message
            remaining_text = remaining_text[implicit_reasoning_end:]
            reasoning_closed = True
        else:
            reasoning = strip_repeated_think_begin(
                remaining_text[reasoning_start:reasoning_end]
            )
            if reasoning.endswith("\n"):
                reasoning = reasoning[:-1]
            if reasoning:
                message["reasoning"] = reasoning
            remaining_text = remaining_text[reasoning_end + len(THINK_END) :]
            if remaining_text.startswith("\n\n"):
                remaining_text = remaining_text[2:]
            reasoning_closed = True

    tool_call_begin = remaining_text.find(TOOL_CALL_BEGIN)
    if tool_call_begin == -1:
        message["content"] = remaining_text
    else:
        tool_calls = parse_tool_calls(remaining_text[tool_call_begin:], tools)
        if tool_calls:
            message["content"] = re.sub(r"\n+$", "", remaining_text[:tool_call_begin])
            message["tool_calls"] = tool_calls
        else:
            message["content"] = remaining_text

    if has_im_end:
        message["finish_reason"] = "tool_calls" if message.get("tool_calls") else "stop"
    elif (
        reasoning_closed
        and not message.get("content")
        and not message.get("tool_calls")
    ):
        message["finish_reason"] = REASONING_END
    if (
        has_assistant_begin
        and not message.get("content")
        and not message.get("reasoning")
        and not message.get("tool_calls")
    ):
        message["content"] = ""
    return message


def normalize_message_tool_calls(message, messages=None):
    """Qwen3.5 does not emit tool call ids, so synthesize ids unique across the trajectory."""
    if not message.get("tool_calls"):
        return message
    used_tool_call_ids = set()
    previous_tool_call_count = 0
    for previous_message in messages or []:
        for tool_call in previous_message.get("tool_calls") or []:
            previous_tool_call_count += 1
            if tool_call.get("id"):
                used_tool_call_ids.add(tool_call["id"])
    for tool_call in message["tool_calls"]:
        if tool_call.get("id"):
            used_tool_call_ids.add(tool_call["id"])

    for tool_call_index, tool_call in enumerate(message["tool_calls"]):
        tool_call["index"] = tool_call_index
        if tool_call.get("id"):
            continue
        tool_call_id_index = previous_tool_call_count + tool_call_index
        function_name = tool_call["function"]["name"]
        tool_call_id = f"functions.{function_name}:{tool_call_id_index}"
        while tool_call_id in used_tool_call_ids:
            tool_call_id_index += 1
            tool_call_id = f"functions.{function_name}:{tool_call_id_index}"
        tool_call["id"] = tool_call_id
        used_tool_call_ids.add(tool_call_id)
    return message


class Qwen3p5ResponseTemplate:
    """Plain text template: the templated prompt is the model's literal response text."""

    @staticmethod
    def match(response_template=None):
        return bool(
            re.match(
                r"Qwen/Qwen3\.[5-8](-|$)",
                (response_template or {}).get("name_or_path") or "",
                re.IGNORECASE,
            )
        )

    def __init__(self, response_template=None, special_tokens=None):
        self.config = response_template or {}
        self.reasoning_end_marker = THINK_END

    def __str__(self):
        return f"{type(self).__name__}({self.config})"

    __repr__ = __str__

    def apply(self, message):
        templated_prompt = ""
        key_path_prompt_mapping = []

        def append_raw(text):
            nonlocal templated_prompt
            templated_prompt += text

        def append_mapped(key_path, text):
            nonlocal templated_prompt
            if not text:
                return
            key_path_prompt_mapping.append(
                dict(
                    key_path=key_path,
                    text_start=len(templated_prompt),
                    text_end=len(templated_prompt) + len(text),
                )
            )
            templated_prompt += text

        finish_reason = message.get("finish_reason")
        is_partial = finish_reason not in COMPLETE_FINISH_REASONS
        reasoning = (
            strip_repeated_think_begin(message["reasoning"])
            if message.get("reasoning")
            else ""
        )
        has_response_body = bool(
            reasoning or message.get("content") or message.get("tool_calls")
        )
        is_pure_reasoning_partial = (
            is_partial
            and finish_reason != REASONING_END
            and not message.get("content")
            and not message.get("tool_calls")
        )

        if reasoning or (
            not has_response_body and is_partial and finish_reason != REASONING_END
        ):
            append_raw(THINK_BEGIN + "\n")
            append_mapped(["reasoning"], reasoning)
            if not is_pure_reasoning_partial:
                append_raw("\n" + THINK_END)
                if message.get("content") or message.get("tool_calls"):
                    append_raw("\n\n")
        append_mapped(["content"], message.get("content"))

        if message.get("tool_calls"):
            if message.get("content"):
                append_raw("\n\n")
            for tool_call_position, tool_call in enumerate(message["tool_calls"]):
                if tool_call_position:
                    append_raw("\n")
                append_raw(
                    TOOL_CALL_BEGIN
                    + "\n"
                    + FUNCTION_BEGIN
                    + tool_call["function"]["name"]
                )
                if "arguments" not in tool_call["function"]:
                    continue
                append_raw(">\n")
                arguments = tool_call["function"]["arguments"]
                parsed_arguments = parse_partial_json_object(arguments)
                is_last_partial_tool_call = (
                    is_partial and tool_call_position == len(message["tool_calls"]) - 1
                )
                if parsed_arguments:
                    for parameter in parsed_arguments["entries"]:
                        append_raw(PARAMETER_BEGIN + parameter["name"])
                        if not parameter["name_complete"]:
                            break
                        append_raw(">\n")
                        if "value" not in parameter:
                            break
                        append_mapped(
                            ["tool_calls", tool_call_position, "function", "arguments"],
                            (
                                _argument_value_to_text(parameter["value"])
                                if parameter["complete"]
                                else parameter["value"]
                            ),
                        )
                        if parameter["complete"]:
                            append_raw("\n" + PARAMETER_END + "\n")
                else:
                    append_mapped(
                        ["tool_calls", tool_call_position, "function", "arguments"],
                        arguments,
                    )

                function_complete = not is_last_partial_tool_call or (
                    parsed_arguments and parsed_arguments["complete"]
                )
                if function_complete:
                    if not parsed_arguments and arguments:
                        append_raw("\n")
                    append_raw(FUNCTION_END)
                if not is_last_partial_tool_call:
                    append_raw("\n" + TOOL_CALL_END)
        return dict(
            templated_prompt=templated_prompt,
            key_path_prompt_mapping=key_path_prompt_mapping,
        )

    def parse(self, text, messages=None, tools=None, finish_reason=None):
        if not text:
            return {}
        message = parse_qwen_response_text(text, tools)
        if finish_reason:
            message["finish_reason"] = (
                "tool_calls" if message.get("tool_calls") else finish_reason
            )
        return normalize_message_tool_calls(message, messages)


def test_qwen3p5_response_template():
    template = Qwen3p5ResponseTemplate()
    tools = [
        dict(
            type="function",
            function=dict(
                name="read_file",
                parameters=dict(
                    type="object",
                    properties=dict(
                        path=dict(type="string"),
                        text=dict(type="string"),
                        limit=dict(type="integer"),
                        flag=dict(type="boolean"),
                        list=dict(type="array"),
                        obj=dict(type="object"),
                    ),
                ),
            ),
        )
    ]

    def assert_equal(actual, expected, label):
        assert (
            actual == expected
        ), f"{label}\n  actual  : {actual!r}\n  expected: {expected!r}"

    # [partial arguments, parameters text following the function name, arguments parsed back
    # from the templated prompt]. An unclosed value stays a JSON string prefix, because only
    # the schema of a closed value tells its real type.
    partial_arguments_cases = [
        ["", "", ""],
        ["{", "", ""],
        ['{"', PARAMETER_BEGIN, '{"'],
        ['{"pa', PARAMETER_BEGIN + "pa", '{"pa'],
        ['{"path"', PARAMETER_BEGIN + "path>\n", '{"path": "'],
        ['{"path":', PARAMETER_BEGIN + "path>\n", '{"path": "'],
        ['{"path": "', PARAMETER_BEGIN + "path>\n", '{"path": "'],
        ['{"path": "/tm', PARAMETER_BEGIN + "path>\n/tm", '{"path": "/tm'],
        [
            '{"path": "/tmp"',
            PARAMETER_BEGIN + "path>\n/tmp\n" + PARAMETER_END + "\n",
            '{"path": "/tmp", ',
        ],
        [
            '{"path": "/tmp",',
            PARAMETER_BEGIN + "path>\n/tmp\n" + PARAMETER_END + "\n",
            '{"path": "/tmp", ',
        ],
        [
            '{"path": "/tmp", "',
            PARAMETER_BEGIN + "path>\n/tmp\n" + PARAMETER_END + "\n" + PARAMETER_BEGIN,
            '{"path": "/tmp", "',
        ],
        [
            '{"path": "/tmp", "limit": 1',
            PARAMETER_BEGIN
            + "path>\n/tmp\n"
            + PARAMETER_END
            + "\n"
            + PARAMETER_BEGIN
            + "limit>\n1",
            '{"path": "/tmp", "limit": "1',
        ],
        # A trailing number may still grow, so it stays unclosed while true, false and null cannot grow.
        ['{"limit": 10', PARAMETER_BEGIN + "limit>\n10", '{"limit": "10'],
        ['{"flag": tr', PARAMETER_BEGIN + "flag>\ntr", '{"flag": "tr'],
        [
            '{"flag": true',
            PARAMETER_BEGIN + "flag>\nTrue\n" + PARAMETER_END + "\n",
            '{"flag": true, ',
        ],
        ['{"list": [1, 2', PARAMETER_BEGIN + "list>\n[1, 2", '{"list": "[1, 2'],
        [
            '{"list": [1, 2]',
            PARAMETER_BEGIN + "list>\n[1, 2]\n" + PARAMETER_END + "\n",
            '{"list": [1, 2], ',
        ],
        ['{"obj": {"a"', PARAMETER_BEGIN + 'obj>\n{"a"', '{"obj": "{\\"a\\"'],
        [
            '{"obj": {"a": 1}',
            PARAMETER_BEGIN + 'obj>\n{"a": 1}\n' + PARAMETER_END + "\n",
            '{"obj": {"a": 1}, ',
        ],
        [
            '{"text": "say \\"hi',
            PARAMETER_BEGIN + 'text>\nsay "hi',
            '{"text": "say \\"hi',
        ],
        [
            '{"text": "line1\\n',
            PARAMETER_BEGIN + "text>\nline1\n",
            '{"text": "line1\\n',
        ],
        ['{"text": "a\\\\', PARAMETER_BEGIN + "text>\na\\", '{"text": "a\\\\'],
        [
            '{"limit": 10}',
            PARAMETER_BEGIN + "limit>\n10\n" + PARAMETER_END + "\n" + FUNCTION_END,
            '{"limit": 10}',
        ],
        ["{}", FUNCTION_END, "{}"],
        # Arguments that are not a JSON object prefix stay verbatim in the function body.
        ['{"text": "raw\nnewline', '{"text": "raw\nnewline', '{"text": "raw\nnewline'],
        ["oops", "oops", "oops"],
    ]
    tool_call_prefix = TOOL_CALL_BEGIN + "\n" + FUNCTION_BEGIN + "read_file>\n"
    for (
        arguments_text,
        parameters_text,
        parsed_arguments_text,
    ) in partial_arguments_cases:
        message = dict(
            role="assistant",
            tool_calls=[
                dict(
                    index=0,
                    type="function",
                    function=dict(name="read_file", arguments=arguments_text),
                )
            ],
        )
        templated_prompt = template.apply(message)["templated_prompt"]
        case_label = f"partial arguments {arguments_text!r}"
        assert_equal(
            templated_prompt, tool_call_prefix + parameters_text, f"apply {case_label}"
        )
        parsed_message = template.parse(templated_prompt, tools=tools)
        assert_equal(
            parsed_message["tool_calls"][0]["function"]["arguments"],
            parsed_arguments_text,
            f"parse {case_label}",
        )
        assert_equal(
            template.apply(parsed_message)["templated_prompt"],
            templated_prompt,
            f"re-apply {case_label}",
        )

    # An unterminated function name means the arguments channel has not started yet.
    open_name_text = TOOL_CALL_BEGIN + "\n" + FUNCTION_BEGIN + "read_fi"
    open_name_message = template.parse(open_name_text, tools=tools)
    assert_equal(
        open_name_message["tool_calls"][0]["function"]["name"],
        "read_fi",
        "open function name",
    )
    assert (
        "arguments" not in open_name_message["tool_calls"][0]["function"]
    ), open_name_message
    assert_equal(
        template.apply(open_name_message)["templated_prompt"],
        open_name_text,
        "re-apply open function name",
    )

    complete_text = (
        THINK_BEGIN
        + "\nthinking\n"
        + THINK_END
        + "\n\nSome content\n\n"
        + TOOL_CALL_BEGIN
        + "\n"
        + FUNCTION_BEGIN
        + "read_file>\n"
        + PARAMETER_BEGIN
        + "path>\n/tmp/a.txt\n"
        + PARAMETER_END
        + "\n"
        + PARAMETER_BEGIN
        + "limit>\n10\n"
        + PARAMETER_END
        + "\n"
        + FUNCTION_END
        + "\n"
        + TOOL_CALL_END
    )
    complete_message = template.parse(complete_text, tools=tools, finish_reason="stop")
    assert_equal(
        complete_message["tool_calls"][0]["function"]["arguments"],
        '{"path": "/tmp/a.txt", "limit": 10}',
        "complete arguments",
    )
    assert_equal(
        complete_message["finish_reason"], "tool_calls", "complete finish_reason"
    )
    assert_equal(
        template.apply(complete_message)["templated_prompt"],
        complete_text,
        "re-apply complete response",
    )

    # Reasoning only continuation, and the reasoning_end boundary.
    assert_equal(
        template.apply(dict(role="assistant", reasoning="think"))["templated_prompt"],
        THINK_BEGIN + "\nthink",
        "apply pure reasoning partial",
    )
    reasoning_end_text = THINK_BEGIN + "\nthink\n" + THINK_END
    assert_equal(
        template.parse(reasoning_end_text).get("finish_reason"),
        REASONING_END,
        "reasoning_end",
    )
    assert_equal(
        template.apply(template.parse(reasoning_end_text))["templated_prompt"],
        reasoning_end_text,
        "re-apply reasoning_end",
    )
    assert Qwen3p5ResponseTemplate.match(dict(name_or_path="Qwen/Qwen3.6-35B-A3B"))
    assert not Qwen3p5ResponseTemplate.match(
        dict(name_or_path="Qwen/Qwen2.5-7B-Instruct")
    )
    return len(partial_arguments_cases) + 4


if __name__ == "__main__":
    print("test_qwen3p5_response_template passed:", test_qwen3p5_response_template())
