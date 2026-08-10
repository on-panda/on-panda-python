#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Response template for the Step-3.5 through Step-3.9 Flash family."""

import re
import mximport

with mximport.inpkg():
    from .qwen3p5 import (
        FUNCTION_BEGIN,
        FUNCTION_END,
        PARAMETER_BEGIN,
        PARAMETER_END,
        Qwen3p5ResponseTemplate,
        THINK_BEGIN,
        THINK_END,
        TOOL_CALL_BEGIN,
        TOOL_CALL_END,
    )


class Step3p5ResponseTemplate(Qwen3p5ResponseTemplate):
    """Step Flash uses Qwen-style XML calls with different response separators."""

    reasoning_content_separator = "\n"
    content_tool_calls_separator = ""
    tool_call_separator = ""

    @staticmethod
    def match(response_template=None):
        return bool(
            re.match(
                r"stepfun-ai/Step-3\.[5-9]-Flash(-|$)",
                (response_template or {}).get("name_or_path") or "",
                re.IGNORECASE,
            )
        )


def test_step3p5_response_template():
    template = Step3p5ResponseTemplate()
    message = {
        "role": "assistant",
        "reasoning": "I should check both cities.",
        "content": "I will check them now.",
        "tool_calls": [
            {
                "type": "function",
                "index": 0,
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "New York City, NY"}',
                },
            },
            {
                "type": "function",
                "index": 1,
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA"}',
                },
            },
        ],
        "finish_reason": "tool_calls",
    }
    expected = (
        f"{THINK_BEGIN}\nI should check both cities.\n{THINK_END}\n"
        f"I will check them now.{TOOL_CALL_BEGIN}\n"
        f"{FUNCTION_BEGIN}get_weather>\n"
        f"{PARAMETER_BEGIN}location>\nNew York City, NY\n{PARAMETER_END}\n"
        f"{FUNCTION_END}\n{TOOL_CALL_END}{TOOL_CALL_BEGIN}\n"
        f"{FUNCTION_BEGIN}get_weather>\n"
        f"{PARAMETER_BEGIN}location>\nSan Francisco, CA\n{PARAMETER_END}\n"
        f"{FUNCTION_END}\n{TOOL_CALL_END}"
    )
    assert template.apply(message)["templated_prompt"] == expected
    assert (
        template.apply(template.parse(expected, finish_reason="tool_calls"))[
            "templated_prompt"
        ]
        == expected
    )

    for version in range(5, 10):
        assert Step3p5ResponseTemplate.match(
            dict(name_or_path=f"stepfun-ai/Step-3.{version}-Flash")
        )
    assert not Step3p5ResponseTemplate.match(
        dict(name_or_path="stepfun-ai/Step-3.4-Flash")
    )
    assert not Step3p5ResponseTemplate.match(
        dict(name_or_path="step3p7-mm-fp8-mtp3-it100")
    )
    return 9


if __name__ == "__main__":
    print("test_step3p5_response_template passed:", test_step3p5_response_template())
