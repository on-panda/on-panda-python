#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onPanda canonical response template, used by the prompt based reasoning correcting model.

Every output channel becomes plain text, so a find-and-replace correction can cut
anywhere and express any channel end. See onPanda `core_design.md` for the format.
"""


def _special_marker(name):
    # Assemble markers from fragments, so no complete marker literal appears in source.
    return "".join(["<|", name, "|>"])


def _call_marker(name):
    return "".join(["<|", "ON_", "PANDA_", name, "|>"])


MARKER_PAIR_SEPARATOR = "\n\n"
TOOL_CALLS_MARKER = _special_marker("tool_calls")
TOOL_RESPONSE_MARKER = _call_marker("TOOL_RESPONSE")
CALL_BEGIN_MARKER = _call_marker("CALL_BEGIN")
# `function.arguments` is the last field, so it runs to the end of the record and may
# contain raw newlines, which a partial JSON prefix fallback needs.
CALL_FIELD_MARKERS = [
    (["type"], _call_marker("CALL_TYPE")),
    (["id"], _call_marker("CALL_ID")),
    (["function", "name"], _call_marker("CALL_NAME")),
    (["function", "arguments"], _call_marker("CALL_ARGUMENTS")),
]
FLATTENED_MESSAGE_KEYS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
)
COMPLETE_FINISH_REASONS = ("stop", "tool_calls")
REASONING_END = "reasoning_end"


def content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    assert all([chunk["type"] == "text" for chunk in content]), content
    return "".join([chunk.get("text", "") for chunk in content])


def _get_by_key_path(data, key_path):
    for key in key_path:
        if key not in data:
            return None
        data = data[key]
    return data


def _set_by_key_path(data, key_path, value):
    for key in key_path[:-1]:
        data = data.setdefault(key, {})
    data[key_path[-1]] = value


def _matched_suffix_length(text, marker):
    """Length of the longest non-empty prefix of `marker` that `text` ends with, 0 when none."""
    for length in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:length]):
            return length
    return 0


def parse_tool_call_records(tool_calls_text):
    """A complete call marker opens a record; a field marker means the key exists."""
    if not tool_calls_text.startswith(CALL_BEGIN_MARKER):
        return []
    records = tool_calls_text[len(CALL_BEGIN_MARKER) :].removeprefix("\n").split(
        "\n" + CALL_BEGIN_MARKER + "\n"
    )
    tool_calls = []
    for tool_call_index, record in enumerate(records):
        tool_call = dict(index=tool_call_index)
        for field_index, (key_path, marker) in enumerate(CALL_FIELD_MARKERS):
            marker_index = record.find(marker)
            if marker_index == -1:
                continue
            value_start = marker_index + len(marker)
            line_end = record.find("\n", value_start)
            is_last_field = field_index == len(CALL_FIELD_MARKERS) - 1
            if is_last_field or line_end == -1:
                value = record[value_start:]
            else:
                value = record[value_start:line_end]
            _set_by_key_path(tool_call, key_path, value)
        tool_calls.append(tool_call)
    return tool_calls


class DefaultResponseTemplate:
    """
    `<|reasoning|>{reasoning}<|reasoning|>\\n\\n<|reasoning|>{content}<|tool_calls|>\\n\\n<|tool_calls|>{tool_calls}<|stop|>`

    A marker means the channel exists, so a content only message stays marker free and
    renders exactly as before this template existed. The reasoning and stop markers come
    from `special_tokens`, because the tokenizer aware correcting model needs them to be
    single tokens of its own tokenizer.
    """

    def __init__(self, response_template=None, special_tokens=None):
        special_tokens = special_tokens or {}
        self.config = response_template or {}
        self.reasoning_marker = special_tokens.get(
            "reasoning", _special_marker("reasoning")
        )
        self.stop_marker = special_tokens.get("stop", _special_marker("stop"))
        self.reasoning_end_marker = (
            self.reasoning_marker + MARKER_PAIR_SEPARATOR + self.reasoning_marker
        )
        self.tool_calls_begin_marker = (
            TOOL_CALLS_MARKER + MARKER_PAIR_SEPARATOR + TOOL_CALLS_MARKER
        )

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
            key_path_prompt_mapping.append(
                dict(
                    key_path=key_path,
                    text_start=len(templated_prompt),
                    text_end=len(templated_prompt) + len(text),
                    channel_start=0,
                    channel_end=len(text),
                )
            )
            templated_prompt += text

        finish_reason = message.get("finish_reason")
        is_partial = finish_reason not in COMPLETE_FINISH_REASONS
        if message.get("reasoning") is not None:
            append_raw(self.reasoning_marker)
            append_mapped(["reasoning"], message["reasoning"])
            is_thinking_open = (
                is_partial
                and finish_reason != REASONING_END
                and not message.get("content")
                and message.get("tool_calls") is None
            )
            if not is_thinking_open:
                append_raw(self.reasoning_end_marker)
        if "content" in message:
            append_mapped(["content"], content_to_text(message["content"]))
        if message.get("tool_calls") is not None:
            append_raw(self.tool_calls_begin_marker)
            for tool_call_index, tool_call in enumerate(message["tool_calls"]):
                append_raw(("\n" if tool_call_index else "") + CALL_BEGIN_MARKER + "\n")
                is_first_field = True
                for key_path, marker in CALL_FIELD_MARKERS:
                    value = _get_by_key_path(tool_call, key_path)
                    if value is None:
                        continue
                    append_raw(("" if is_first_field else "\n") + marker)
                    append_mapped(["tool_calls", tool_call_index] + key_path, value)
                    is_first_field = False
        if not is_partial:
            append_raw(self.stop_marker)
        return dict(
            templated_prompt=templated_prompt,
            key_path_prompt_mapping=key_path_prompt_mapping,
        )

    def parse(self, text, messages=None, tools=None, finish_reason=None):
        message = dict(role="assistant")
        stop_index = text.find(self.stop_marker)
        if stop_index != -1:
            # A correcting model may keep generating after the stop marker, drop that tail.
            text = text[:stop_index]
            finish_reason = "stop"
        if text.startswith(self.reasoning_marker):
            reasoning_text = text[len(self.reasoning_marker) :]
            reasoning_end_index = reasoning_text.find(self.reasoning_marker)
            if reasoning_end_index == -1:
                message["reasoning"] = reasoning_text
                if finish_reason:
                    message["finish_reason"] = finish_reason
                return message
            message["reasoning"] = reasoning_text[:reasoning_end_index]
            text = reasoning_text[reasoning_end_index + len(self.reasoning_marker) :]
            # A cut or a replacement may stop inside the closing marker pair, so consume the pair
            # field by field. Thinking ended as soon as its first marker showed up.
            if text.startswith(MARKER_PAIR_SEPARATOR):
                text = text[len(MARKER_PAIR_SEPARATOR) :]
                if text.startswith(self.reasoning_marker):
                    text = text[len(self.reasoning_marker) :]
            elif MARKER_PAIR_SEPARATOR.startswith(text):
                text = ""
            if not text:
                finish_reason = finish_reason or REASONING_END
        tool_calls_index = text.find(self.tool_calls_begin_marker)
        if tool_calls_index != -1:
            message["content"] = text[:tool_calls_index]
            message["tool_calls"] = parse_tool_call_records(
                text[tool_calls_index + len(self.tool_calls_begin_marker) :]
            )
        else:
            # The channel opens at the first complete marker, even if the pair is still partial.
            opened_length = _matched_suffix_length(text, self.tool_calls_begin_marker)
            if opened_length >= len(TOOL_CALLS_MARKER):
                message["content"] = text[: len(text) - opened_length]
                message["tool_calls"] = []
            else:
                message["content"] = text
        if finish_reason:
            message["finish_reason"] = finish_reason
            if finish_reason == "stop" and message.get("tool_calls"):
                message["finish_reason"] = "tool_calls"
        return message


def test_default_response_template():
    template = DefaultResponseTemplate()
    reasoning_marker = template.reasoning_marker
    reasoning_end = template.reasoning_end_marker
    tool_calls_begin = template.tool_calls_begin_marker
    stop = template.stop_marker
    call_type, call_id, call_name, call_arguments = [
        marker for _, marker in CALL_FIELD_MARKERS
    ]

    def assert_equal(actual, expected, label):
        assert (
            actual == expected
        ), f"{label}\n  actual  : {actual!r}\n  expected: {expected!r}"

    # Canonical messages: apply matches the expected text, and re-applying the parsed
    # message reproduces it, which is the exactness the correcting round trip relies on.
    canonical_cases = [
        (dict(role="assistant", content="hi"), "hi"),
        (dict(role="assistant", content="hi", finish_reason="stop"), "hi" + stop),
        (
            dict(role="assistant", reasoning="think"),
            reasoning_marker + "think",
        ),
        (
            dict(
                role="assistant",
                reasoning="think",
                content="",
                finish_reason=REASONING_END,
            ),
            reasoning_marker + "think" + reasoning_end,
        ),
        (
            dict(
                role="assistant",
                reasoning="think",
                content="answer",
                finish_reason="stop",
            ),
            reasoning_marker + "think" + reasoning_end + "answer" + stop,
        ),
        (dict(role="assistant", content="", tool_calls=[]), tool_calls_begin),
        (
            # Thinking ended and the tool call channel opened, but nothing is written yet.
            dict(role="assistant", reasoning="think", content="", tool_calls=[]),
            reasoning_marker + "think" + reasoning_end + tool_calls_begin,
        ),
        (
            dict(
                role="assistant",
                content="calling",
                tool_calls=[
                    dict(
                        index=0,
                        type="function",
                        id="functions.read_file:0",
                        function=dict(
                            name="read_file", arguments='{"path": "/tmp/a.txt"}'
                        ),
                    )
                ],
                finish_reason="tool_calls",
            ),
            "calling"
            + tool_calls_begin
            + CALL_BEGIN_MARKER
            + "\n"
            + call_type
            + "function"
            + "\n"
            + call_id
            + "functions.read_file:0"
            + "\n"
            + call_name
            + "read_file"
            + "\n"
            + call_arguments
            + '{"path": "/tmp/a.txt"}'
            + stop,
        ),
        (
            # The arguments channel has not started, so the function name is still open.
            dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(index=0, type="function", function=dict(name="read_fi"))
                ],
            ),
            tool_calls_begin
            + CALL_BEGIN_MARKER
            + "\n"
            + call_type
            + "function"
            + "\n"
            + call_name
            + "read_fi",
        ),
        (
            # Raw newlines inside partial arguments stay verbatim, and multi call records
            # are separated by a marker instead of a newline.
            dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(
                        index=0,
                        function=dict(name="a", arguments='{"text": "line1\nline2"}'),
                    ),
                    dict(index=1, function=dict(name="b", arguments='{"x": ')),
                ],
            ),
            tool_calls_begin
            + CALL_BEGIN_MARKER
            + "\n"
            + call_name
            + "a"
            + "\n"
            + call_arguments
            + '{"text": "line1\nline2"}'
            + "\n"
            + CALL_BEGIN_MARKER
            + "\n"
            + call_name
            + "b"
            + "\n"
            + call_arguments
            + '{"x": ',
        ),
    ]
    for message, templated_prompt in canonical_cases:
        label = f"canonical {message}"
        assert_equal(
            template.apply(message)["templated_prompt"],
            templated_prompt,
            f"apply {label}",
        )
        parsed_message = template.parse(templated_prompt)
        assert_equal(
            template.apply(parsed_message)["templated_prompt"],
            templated_prompt,
            f"re-apply {label}",
        )

    # Truncation cases: a cut inside any channel must keep that channel's identity.
    reasoning_cut = template.parse(reasoning_marker + "think pre")
    assert_equal(reasoning_cut.get("reasoning"), "think pre", "cut inside reasoning")
    assert "content" not in reasoning_cut, reasoning_cut
    assert_equal(
        template.parse(reasoning_marker + "think" + reasoning_marker).get(
            "finish_reason"
        ),
        REASONING_END,
        "replacement is a single reasoning marker",
    )
    # A replacement may cut the closing marker pair short to open the next channel.
    channel_switch = template.parse(
        reasoning_marker
        + "think"
        + reasoning_marker
        + MARKER_PAIR_SEPARATOR
        + TOOL_CALLS_MARKER
    )
    assert_equal(
        {key: channel_switch[key] for key in ("reasoning", "content", "tool_calls")},
        dict(reasoning="think", content="", tool_calls=[]),
        "replacement switches from reasoning to tool calls",
    )
    arguments_cut = template.parse(
        tool_calls_begin
        + CALL_BEGIN_MARKER
        + "\n"
        + call_name
        + "a"
        + "\n"
        + call_arguments
        + '{"p": "/tm'
    )
    assert_equal(
        arguments_cut["tool_calls"][0]["function"]["arguments"],
        '{"p": "/tm',
        "cut inside arguments",
    )
    assert_equal(
        template.parse("done" + stop + "junk").get("content"), "done", "tail after stop"
    )

    # key_path_prompt_mapping locates every channel inside the templated prompt.
    apply_result = template.apply(canonical_cases[4][0])
    assert_equal(
        [mapping["key_path"] for mapping in apply_result["key_path_prompt_mapping"]],
        [["reasoning"], ["content"]],
        "key_path_prompt_mapping key paths",
    )
    for mapping in apply_result["key_path_prompt_mapping"]:
        assert_equal(
            apply_result["templated_prompt"][
                mapping["text_start"] : mapping["text_end"]
            ],
            canonical_cases[4][0][mapping["key_path"][0]],
            "key_path_prompt_mapping range",
        )
    return len(canonical_cases)


if __name__ == "__main__":
    print("test_default_response_template passed:", test_default_response_template())
