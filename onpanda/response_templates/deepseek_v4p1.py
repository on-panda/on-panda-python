#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek V4.1 Flash uses spaced DSML tags and the calls block."""

import re

import mximport

with mximport.inpkg():
    from .deepseek_v4 import DeepSeekV4ResponseTemplate


class DeepSeekV4p1ResponseTemplate(DeepSeekV4ResponseTemplate):
    default_name_or_path = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    tool_calls_block_name = " calls"
    tool_call_tag_name = " invoke"
    parameter_tag_name = " parameter"

    @staticmethod
    def match(response_template=None):
        return bool(
            re.match(
                r"(?:deepseek-ai/)?DeepSeek-V4\.1-Flash(?:-|$)",
                (response_template or {}).get("name_or_path") or "",
                re.IGNORECASE,
            )
        )
