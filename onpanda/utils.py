#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:23:49 2024

@author: DIYer22
"""
HASH_TEMPLATE_PREFIX = "<|hash|>"
HASH_TEMPLATE_REGEX = r"^<\|hash\|>([A-Za-z0-9+\/=]+)$"
HASH_MAP_LENGTH = 16000
RESPONSE_ROLES = ["assistant"]

true = True
false = False
null = None


def messages_to_panda_tree(msgs, uuid=None):
    panda_tree = {
        "dialogs": {
            "1": {
                "messages": msgs,
                "annotate": {
                    "is_good": None,  # Open Data Pannel to annotate
                },
            }
        }
    }
    if uuid is not None:
        panda_tree["uuid"] = uuid
    return panda_tree


def remove_msgs_after_last_response_role(messages, response_roles=None):
    """
    Keep messages up to and including the last message whose role is in response_roles.
    """
    if response_roles is None:
        response_roles = RESPONSE_ROLES
    last_response_idx = -1
    for msg_idx, msg in enumerate(messages):
        if msg["role"] in response_roles:
            last_response_idx = msg_idx
    assert last_response_idx >= 0, messages
    return messages[: last_response_idx + 1]


def get_content_by_path_keys(data, path_keys):
    target = data
    for key in path_keys:
        target = target[key]
    return target


JSON_TRUNCATE_LIMIT = 64 * 1024
JSON_TRUNCATE_KEEP = 1024
JSON_TRUNCATE_PLACEHOLDER = (
    "\n\n\n.......................\n{reason}\n.................\n\n\n"
)


def slim_json_strings(
    data,
    *,
    limit=JSON_TRUNCATE_LIMIT,
    keep=JSON_TRUNCATE_KEEP,
    placeholder=JSON_TRUNCATE_PLACEHOLDER,
):
    """Recursively trim strings that exceed ``limit`` characters."""
    if isinstance(data, dict):
        return {
            key: slim_json_strings(
                value, limit=limit, keep=keep, placeholder=placeholder
            )
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            slim_json_strings(value, limit=limit, keep=keep, placeholder=placeholder)
            for value in data
        ]
    if isinstance(data, str):
        if len(data) <= limit:
            return data
        head = data[:keep]
        tail = data[-keep:]
        reason = f"omitted {len(data) - 2 * keep} chars"
        return head + placeholder.format(reason=reason) + tail
    return data


if __name__ == "__main__":
    pass
    image_url_msg_example = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图中左侧的 “v” 是由什么形状构成？\n"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://docs.vllm.ai/en/latest/_static/vllm-logo-text-light.png"
                    },
                },
            ],
        }
    ]
    panda_tree_example = {
        "dialogs": {
            "1": {
                "messages": [
                    {"role": "user", "content": "🍓How many `R` in strawberry?"}
                ],
                "operations": [
                    {
                        "time": 1734883835911,
                        "parent": "1",
                        "chat_config": {
                            "model": "vllm-model",
                            "max_tokens": 1024,
                            "temperature": 0.5,
                        },
                        "common_prefix_length": -1,
                        "operator": "generate_new",
                        "is_new_generated": true,
                        "on_policy": true,
                    }
                ],
            }
        },
        "version": "1.0",
        "uuid": "2024-12-23_00-10-35.908",
        "hash_map": {},
        "deleted_dialogs": {},
        "update_time": 1734883863852,
    }
    panda_tree = messages_to_panda_tree(image_url_msg_example)
    from pprint import pp

    pp(panda_tree)
