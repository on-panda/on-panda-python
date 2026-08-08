#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan a JSON object prefix, such as tool call arguments a model is still writing.

Ported from onPanda vue `src/utils/partialJsonUtils.js`, keep both sides identical.
"""

import json
import re

JSON_SPACES = " \t\n\r"
GROWABLE_LITERALS = ("true", "false", "null")
DANGLING_ESCAPE_REGEX = re.compile(r"(?:^|[^\\])(?:\\\\)*(\\(?:u[0-9a-fA-F]{0,3})?)$")


def _skip_json_spaces(text, cursor):
    while cursor < len(text) and text[cursor] in JSON_SPACES:
        cursor += 1
    return cursor


def _find_json_string_end(text, quote_start):
    # Index of the closing quote, -1 when the string is still open.
    cursor = quote_start + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 1
        elif text[cursor] == '"':
            return cursor
        cursor += 1
    return -1


def _find_json_value_end(text, value_start):
    # Index right after the value, -1 when the value is still growing.
    if text[value_start] == '"':
        string_end = _find_json_string_end(text, value_start)
        return -1 if string_end == -1 else string_end + 1
    if text[value_start] in "[{":
        depth = 0
        cursor = value_start
        while cursor < len(text):
            if text[cursor] == '"':
                string_end = _find_json_string_end(text, cursor)
                if string_end == -1:
                    return -1
                cursor = string_end
            elif text[cursor] in "[{":
                depth += 1
            elif text[cursor] in "]}":
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        return -1
    literal_end = re.search(r"[\s,\]}]", text[value_start:])
    if literal_end:
        return value_start + literal_end.start()
    # A number prefix such as '1' can still grow into '10', while a spelled out literal cannot grow.
    return len(text) if text[value_start:] in GROWABLE_LITERALS else -1


def _decode_json_string_prefix(raw_text):
    # A prefix can stop inside an escape such as '\\u12', which is not decodable yet.
    dangling_escape = DANGLING_ESCAPE_REGEX.search(raw_text)
    if dangling_escape:
        raw_text = raw_text[: -len(dangling_escape.group(1))]
    try:
        return json.loads(f'"{raw_text}"')
    except ValueError:
        return None


def parse_partial_json_object(text=""):
    """
    entry["value"] is absent before the value starts, the partial value text while the
    value grows, and the parsed JSON value once entry["complete"] is True.
    Return None when the text cannot be a prefix of a JSON object.
    """
    entries = []
    cursor = _skip_json_spaces(text, 0)
    if cursor == len(text):
        return dict(entries=entries, complete=False)
    if text[cursor] != "{":
        return None
    cursor = _skip_json_spaces(text, cursor + 1)
    while cursor < len(text):
        if text[cursor] == "}":
            if _skip_json_spaces(text, cursor + 1) != len(text):
                return None
            return dict(entries=entries, complete=True)
        if text[cursor] != '"':
            return None
        name_end = _find_json_string_end(text, cursor)
        name = _decode_json_string_prefix(
            text[cursor + 1 : len(text) if name_end == -1 else name_end]
        )
        if name is None:
            return None
        entry = dict(name=name, name_complete=name_end != -1, complete=False)
        entries.append(entry)
        if name_end == -1:
            return dict(entries=entries, complete=False)
        cursor = _skip_json_spaces(text, name_end + 1)
        if cursor == len(text):
            return dict(entries=entries, complete=False)
        if text[cursor] != ":":
            return None
        cursor = _skip_json_spaces(text, cursor + 1)
        if cursor == len(text):
            return dict(entries=entries, complete=False)
        value_end = _find_json_value_end(text, cursor)
        if value_end == -1:
            if text[cursor] == '"':
                partial_value = _decode_json_string_prefix(text[cursor + 1 :])
            else:
                partial_value = text[cursor:]
            if partial_value is None:
                return None
            entry["value"] = partial_value
            return dict(entries=entries, complete=False)
        try:
            entry["value"] = json.loads(text[cursor:value_end])
        except ValueError:
            return None
        entry["complete"] = True
        cursor = _skip_json_spaces(text, value_end)
        if cursor == len(text):
            return dict(entries=entries, complete=False)
        if text[cursor] == "}":
            if _skip_json_spaces(text, cursor + 1) != len(text):
                return None
            return dict(entries=entries, complete=True)
        if text[cursor] != ",":
            return None
        cursor = _skip_json_spaces(text, cursor + 1)
    return dict(entries=entries, complete=False)
