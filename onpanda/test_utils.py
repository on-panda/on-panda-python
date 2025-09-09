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
