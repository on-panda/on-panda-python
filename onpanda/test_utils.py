def build_test_tokenizer(name_or_path="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"):
    try:
        from .token_level_supervision_utils import _from_pretrained_local_first
    except ImportError:
        from token_level_supervision_utils import _from_pretrained_local_first

    return _from_pretrained_local_first(name_or_path, use_fast=True)


def get_test_rejected_msgs1():
    rejected_msgs1 = [
        {"role": "user", "content": "Name three kinds of fruit:"},
        {
            "role": "assistant",
            "content": "Apple, potato, banana.",
            "finish_reason": "stop",
            "token_level": {
                "chosen_text": " orange",
                "rejected_text": " potato",
                "chosen_text_unicode_range": [6, 13],  # " potato" start at index 6
                "rejected_text_unicode_range": [6, 13],
                "version": "1.0",
                "chosen_dialog_key": 2,
                "rejected_dialog_key": 1,
                "rejected_finish_reason": "stop",
            },
        },
    ]
    far_text_gt1 = "<|fim_pad|> potato<|fim_pad|>0<|fim_pad|> orange<|fim_pad|>"
    return rejected_msgs1, far_text_gt1


def get_test_reasoning_tool_calls_msgs():
    """A rejected reasoning tool call response, whose thinking picks the wrong file path."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file and return its content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                        "limit": {"type": "integer", "description": "Max line number"},
                    },
                    "required": ["path"],
                },
            },
        }
    ]
    rejected_msgs = [
        {"role": "user", "content": "Read the first 10 lines of /tmp/a.txt for me."},
        {
            "role": "assistant",
            "reasoning": "The user wants /tmp/a.txt, so I should read /tmp/b.txt with limit 10.",
            "content": "",
            "tool_calls": [
                {
                    "index": 0,
                    "type": "function",
                    "id": "functions.read_file:0",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/b.txt", "limit": 10}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
        },
    ]
    return rejected_msgs, tools
