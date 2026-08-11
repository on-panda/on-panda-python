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


def _decode_json_string_prefix(raw_text, allow_dangling_escape):
    # A prefix can stop inside an escape such as '\\u12', which is not decodable yet.
    dangling_escape = (
        DANGLING_ESCAPE_REGEX.search(raw_text) if allow_dangling_escape else None
    )
    raw_end = len(raw_text)
    if dangling_escape:
        raw_end -= len(dangling_escape.group(1))
    try:
        value = json.loads(f'"{raw_text[:raw_end]}"')
    except ValueError:
        return None
    offsets = [0]
    cursor = 0
    while cursor < raw_end:
        if raw_text[cursor] != "\\":
            cursor += 1
            offsets.append(cursor)
            continue
        escape_end = cursor + 2
        if raw_text[cursor + 1] == "u":
            escape_end = cursor + 6
            code_unit = int(raw_text[cursor + 2 : escape_end], 16)
            if (
                0xD800 <= code_unit <= 0xDBFF
                and raw_text[escape_end : escape_end + 2] == "\\u"
                and escape_end + 6 <= raw_end
                and 0xDC00
                <= int(raw_text[escape_end + 2 : escape_end + 6], 16)
                <= 0xDFFF
            ):
                escape_end += 6
        decoded_escape = json.loads(f'"{raw_text[cursor:escape_end]}"')
        offsets.extend([escape_end] * len(decoded_escape))
        cursor = escape_end
    return dict(value=value, offsets=offsets)


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
        name_start = cursor + 1
        name_end = _find_json_string_end(text, cursor)
        decoded_name = _decode_json_string_prefix(
            text[name_start : len(text) if name_end == -1 else name_end],
            name_end == -1,
        )
        if decoded_name is None:
            return None
        name_offsets = [name_start + offset for offset in decoded_name["offsets"]]
        entry = dict(
            name=decoded_name["value"],
            name_complete=name_end != -1,
            name_start=name_start,
            name_end=name_offsets[-1],
            name_offsets=name_offsets,
            complete=False,
        )
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
                decoded_value = _decode_json_string_prefix(
                    text[cursor + 1 :], allow_dangling_escape=True
                )
                partial_value = (
                    None if decoded_value is None else decoded_value["value"]
                )
            else:
                partial_value = text[cursor:]
            if partial_value is None:
                return None
            entry["value"] = partial_value
            entry["value_start"] = cursor + (text[cursor] == '"')
            if text[cursor] == '"':
                entry["value_offsets"] = [
                    cursor + 1 + offset for offset in decoded_value["offsets"]
                ]
                entry["value_end"] = entry["value_offsets"][-1]
            else:
                entry["value_end"] = len(text)
            return dict(entries=entries, complete=False)
        entry["value_start"] = cursor + (text[cursor] == '"')
        entry["value_end"] = value_end - (text[cursor] == '"')
        if text[cursor] == '"':
            decoded_value = _decode_json_string_prefix(
                text[cursor + 1 : value_end - 1], allow_dangling_escape=False
            )
            if decoded_value is None:
                return None
            entry["value"] = decoded_value["value"]
            entry["value_offsets"] = [
                cursor + 1 + offset for offset in decoded_value["offsets"]
            ]
        else:
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
