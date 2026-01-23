#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:23:49 2024

@author: DIYer22
"""
HASH_TEMPLATE_PREFIX = "<|hash|>"
HASH_TEMPLATE_REGEX = r"^<\|hash\|>([A-Za-z0-9+\/=]+)$"
HASH_MAP_LENGTH = 16000

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
