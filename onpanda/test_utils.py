def build_test_tokenizer(name_or_path="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=True,
            local_files_only=True,
        )
    except (OSError, ValueError):
        # Smaller model that ships with Transformers – guarantees the
        # example runs even without the Llama‑3 files.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer


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
    ntp_as_correcting_text_gt1 = (
        "<|fim_pad|> potato<|fim_pad|>0<|fim_pad|> orange<|fim_pad|>"
    )
    return rejected_msgs1, ntp_as_correcting_text_gt1
