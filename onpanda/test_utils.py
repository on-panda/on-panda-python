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
