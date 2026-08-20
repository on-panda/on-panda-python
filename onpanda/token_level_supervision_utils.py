from typing import List, Dict, Tuple

STOP_TOKEN_DEFAULT = "<|stop|>"  # legacy placeholder for tokenizers without EOS


def _is_tokens_reversible(tokens: List[int], tokenizer) -> bool:
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=False) == tokens


def _minimal_reversible_patch(
    tokens: List[int],
    center_idx: int,
    tokenizer,
) -> Tuple[int, int]:
    """
    Return the *smallest* contiguous slice [start, end) that

    • contains ``center_idx`` and
    • round‑trips through decode‑then‑encode unchanged, i.e.
      ``tokens[start:end] == tokenizer.encode(
           tokenizer.decode(tokens[start:end], skip_special_tokens=False),
           add_special_tokens=False
      )``

    This is the formal definition of a *patch* in the docstring.
    """
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens, add_special_tokens=False)
    n = len(tokens)
    # First try the trivial 1‑token span; for most Latin text it succeeds.
    start, end = center_idx, center_idx + 1

    if _is_tokens_reversible(tokens[start:end], tokenizer):
        return start, end
    for token_num in range(1, n + 1):
        for bias in range(token_num):
            start = center_idx - bias
            end = start + token_num
            if start >= 0 and end <= n:
                if _is_tokens_reversible(tokens[start:end], tokenizer):
                    return start, end

    # Fallback: the whole sequence (should never happen in normal text).
    return 0, n


def compute_token_level_supervision(
    *,
    rejected_content: str,
    chosen_content: str,
    tokenizer=None,
    max_chosen_tokens=1,
    STOP_TOKEN=None,
) -> Dict:
    """
    Compute token‑level supervision signals.

    Parameters
    ----------
    chosen_content / rejected_content : str
        The two candidate strings – *chosen* is forked from *rejected*.
    tokenizer : PreTrainedTokenizerBase, optional
    max_chosen_tokens : int
        Maximum number of chosen tokens to keep after the fork. The patch may extend past this
        limit when required to produce a reversible tokenizer boundary.
    STOP_TOKEN : int or str, optional
        Explicit stop token id. When omitted, use ``tokenizer.eos_token_id`` when available and
        fall back to ``STOP_TOKEN_DEFAULT`` for tokenizers without an EOS token.

    Returns
    -------
    dict  with keys

        fork_token_idx : int
            Index of the first diverging token position.
        chosen_token_id / rejected_token_id : int or str
            Token-IDs at that fork position, or the legacy stop placeholder when no EOS exists.
        max_chosen_tokens : int
            The requested maximum chosen patch length.
        chosen_token_ids : list[int or str]
            Token-IDs in the chosen patch, including the stop token when the chosen text ends.
            Tokenizers without an EOS use the legacy ``STOP_TOKEN_DEFAULT`` placeholder.
        chosen_tokens_include_stop : bool
            Whether ``chosen_token_ids`` ends with the resolved stop token.
        chosen_content / rejected_content : list[dict]
            Each list is ``[pre_patch, patch, post_patch]`` where every
            element is a dict ``{"tokens": List[int], "ignore_loss": bool}``.
            The stop token is represented out of band after the content text. Its loss flags
            reuse ``content[-1]``'s ``ignore_loss`` and optional ``rejected_loss``; the final
            chunk may be empty or contain text.
            • For *chosen*, the patch always receives loss (``ignore_loss=False``).
            • For *rejected*, the patch receives rejected loss when it is a *single* token or the
              stop token itself is the fork target. When the patch spans > 1 actual token we set
              ``ignore_loss=True`` for safety, because a many-token rejected is harmful.
    """
    tokenizer = build_tokenizer(tokenizer)

    stop_token_id = STOP_TOKEN
    if stop_token_id is None:
        stop_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(stop_token_id, (list, tuple)):
            stop_token_id = stop_token_id[0] if stop_token_id else None
        if stop_token_id is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                eos_tokens = tokenizer.encode(eos_token, add_special_tokens=False)
                if len(eos_tokens) == 1:
                    stop_token_id = eos_tokens[0]
        if stop_token_id is None:
            stop_token_id = STOP_TOKEN_DEFAULT

    tok_ch = tokenizer.encode(chosen_content, add_special_tokens=False)
    tok_rj = tokenizer.encode(rejected_content, add_special_tokens=False)

    # 1️⃣ Find the fork position.
    max_common = min(len(tok_ch), len(tok_rj))
    fork_idx = next(
        (i for i in range(max_common) if tok_ch[i] != tok_rj[i]), max_common
    )
    # assert not (
    #     max_common == fork_idx and len(tok_ch) == len(tok_rj)
    # ), f"Chosen and rejected is same! {chosen_content} == {rejected_content}"
    # warning
    if max_common == fork_idx and len(tok_ch) == len(tok_rj):
        print(
            f"Warning: Chosen and rejected are the same! \n  chosen:...{chosen_content[-35:]}\nrejected:...{rejected_content[-35:]}"
        )

    chosen_ends = fork_idx >= len(tok_ch)
    rejected_ends = fork_idx >= len(tok_rj)
    chosen_token_id = stop_token_id if chosen_ends else tok_ch[fork_idx]
    rejected_token_id = stop_token_id if rejected_ends else tok_rj[fork_idx]

    # 2️⃣ Locate minimal reversible patches around the diverging tokens.
    if chosen_ends:
        ch_s = ch_e = len(tok_ch)
    else:
        ch_s, ch_e = _minimal_reversible_patch(tok_ch, fork_idx, tokenizer)
        # Keep ch_s fixed: tokenizer boundaries may move it before fork_idx.
        target_end = min(len(tok_ch), fork_idx + max_chosen_tokens)

        ch_e = next(
            (
                end
                for end in range(max(ch_e, target_end), len(tok_ch) + 1)
                if _is_tokens_reversible(tok_ch[ch_s:end], tokenizer)
            ),
            len(tok_ch),
        )

    if rejected_ends:
        rj_s = rj_e = len(tok_rj)
    else:
        rj_s, rj_e = _minimal_reversible_patch(tok_rj, fork_idx, tokenizer)

    def build_chunks(
        tokens: List[int],
        s: int,
        e: int,
        ignore_patch_loss: bool = False,
        rejected_loss=None,
        is_fork_on_stop=False,
    ):
        """
        Helper – split *tokens* into three segments and label whether
        cross‑entropy loss should be computed (``ignore_loss=False``) or
        masked away (``ignore_loss=True``).
        """
        pre = {
            "type": "text",
            "text": tokenizer.decode(tokens[:s], skip_special_tokens=False),
            "ignore_loss": True,
        }
        patch = {
            "type": "text",
            "text": tokenizer.decode(tokens[s:e], skip_special_tokens=False),
            "ignore_loss": ignore_patch_loss,
            "tokens": tokens[s:e],
        }
        if rejected_loss:
            patch["rejected_loss"] = True
        post = {
            "type": "text",
            "text": tokenizer.decode(tokens[e:], skip_special_tokens=False),
            "ignore_loss": True,
        }
        if is_fork_on_stop:
            return [pre, patch]
        return [pre, patch, post]

    chosen_chunks = build_chunks(tok_ch, ch_s, ch_e, is_fork_on_stop=chosen_ends)
    # Apply negative loss to one-token patches and when stop itself is the rejected fork target.
    rejected_chunks = build_chunks(
        tok_rj,
        rj_s,
        rj_e,
        ignore_patch_loss=(rj_e - rj_s != 1) and not rejected_ends,
        rejected_loss=True,
        is_fork_on_stop=rejected_ends,
    )
    # set chosen_text and rejected_text
    chosen_text = next(
        chunk for chunk in chosen_chunks if not chunk.get("ignore_loss")
    )["text"]
    rejected_text = next(
        chunk for chunk in rejected_chunks if chunk.get("rejected_loss")
    )["text"]
    chosen_tokens_include_stop = ch_e == len(tok_ch)
    chosen_token_ids = list(tok_ch[ch_s:ch_e])
    if chosen_tokens_include_stop and (
        not chosen_token_ids or chosen_token_ids[-1] != stop_token_id
    ):
        chosen_token_ids.append(stop_token_id)
    return {
        "fork_token_idx": fork_idx,
        "chosen_token_id": chosen_token_id,
        "rejected_token_id": rejected_token_id,
        "max_chosen_tokens": max_chosen_tokens,
        "chosen_token_ids": chosen_token_ids,
        "chosen_tokens_include_stop": chosen_tokens_include_stop,
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
        "chosen_text_unicode_range": [
            len(chosen_chunks[0]["text"]),
            len(chosen_chunks[1]["text"]),
        ],
        "rejected_text_unicode_range": [
            len(rejected_chunks[0]["text"]),
            len(rejected_chunks[1]["text"]),
        ],
        "chosen_content": chosen_chunks,
        "rejected_content": rejected_chunks,
    }


def apply_ignore_unicode_loss_mask_to_content(mask, content_str):
    previous_ignore_state = mask[0]
    previous_end_idx = 0
    content_patchs = []
    for idx, state in enumerate(list(mask) + ["add last patch finally"]):
        if state != previous_ignore_state:
            patch = dict(
                text=content_str[previous_end_idx:idx],
                ignore_loss=previous_ignore_state,
                type="text",
            )
            content_patchs.append(patch)
            previous_end_idx = idx
            previous_ignore_state = state
    return content_patchs


class UnicodeTokenizer:
    def __init__(self):
        self.name_or_path = "onpanda.UnicodeTokenizer"

    def encode(self, string, **kwargs):
        return [ord(c) for c in list(string)]

    def decode(self, tokens, **kwargs):
        return "".join([chr(i) for i in tokens])

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        import json

        chatml = json.dumps(messages, indent=2, ensure_ascii=False)
        if tokenize:
            return self.encode(chatml)
        return chatml


unicode_tokenizer = UnicodeTokenizer()


class UTF8Tokenizer:
    def __init__(self):
        self.name_or_path = "onpanda.UTF8Tokenizer"

    def encode(self, string, **kwargs):
        return list(str(string).encode("utf-8"))

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        import json

        chatml = json.dumps(messages, indent=2, ensure_ascii=False)
        if tokenize:
            return self.encode(chatml)
        return chatml


utf8_tokenizer = UTF8Tokenizer()


def _from_pretrained_local_first(name_or_path, **kwargs):
    from transformers import AutoTokenizer

    try:
        local_kwargs = dict(kwargs, local_files_only=True)
        return AutoTokenizer.from_pretrained(name_or_path, **local_kwargs)
    except (OSError, ValueError):
        return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def build_tokenizer(tokenizer=None):
    if not tokenizer or tokenizer == "utf8_tokenizer":
        return utf8_tokenizer
    if tokenizer == "unicode_tokenizer":
        return unicode_tokenizer
    if isinstance(tokenizer, str):
        return _from_pretrained_local_first(tokenizer)
    return tokenizer


# ----------------------------------------------------------------------
# ------------------------------ TESTS ---------------------------------
# ----------------------------------------------------------------------


def _patch_len(result: Dict, which: str, tokenizer) -> int:
    """Utility to grab the token length of the second chunk (the patch)."""
    return len(
        tokenizer.encode(
            result[f"{which}_content"][1]["text"], add_special_tokens=False
        )
    )


def test_one_token_align_one_patch(tok):
    """Patches are single tokens."""
    first_res = compute_token_level_supervision(
        chosen_content="I love cats.", rejected_content="I love dogs.", tokenizer=tok
    )
    assert _patch_len(first_res, "chosen", tok) == 1
    assert _patch_len(first_res, "rejected", tok) == 1
    # Negative loss **is** applied to the rejected patch.

    assert first_res["rejected_content"][1]["ignore_loss"] is False
    g()

    second_res = compute_token_level_supervision(
        chosen_content="Answer is中国",
        rejected_content="Answer is Chinese",
        tokenizer=tok,
    )
    g()
    # assert len(res["chosen_content"][1]["text"]) > 1
    return first_res, second_res


def test_many_token_align_one_patch(tok):
    """Chosen patch >1 token, rejected patch 1 token."""
    many_token_align_one_patch_res = compute_token_level_supervision(
        chosen_content="I like '🥢'", rejected_content="I like '🥄'", tokenizer=tok
    )
    g()
    assert _patch_len(many_token_align_one_patch_res, "chosen", tok) > 1
    assert _patch_len(many_token_align_one_patch_res, "rejected", tok) > 1
    # Negative loss still applies (rejected patch single token).
    assert many_token_align_one_patch_res["rejected_content"][1]["ignore_loss"] is True
    return many_token_align_one_patch_res


def test_many_token_align_many_patch(tok):
    """' 🥢' == [11162, 98, 95]"""
    many_token_align_many_patch_res = compute_token_level_supervision(
        chosen_content="prefix 🥢subfix",  # with space
        rejected_content="prefix🥢subfix",  # without space
        tokenizer=tok,
    )
    rejected_patch = many_token_align_many_patch_res["rejected_content"][1]
    g()
    if len(rejected_patch["tokens"]) > 1:
        assert rejected_patch[
            "ignore_loss"
        ], "if rejected_patch include many tokens, ignore loss. Because a many‑token rejected loss is harmful."
    return many_token_align_many_patch_res


def test_fork_token_is_stop_token(tok):
    chosen_stop_res = compute_token_level_supervision(
        chosen_content="prefix",
        rejected_content="prefix subfix",
        tokenizer=tok,
    )
    rejected_stop_res = compute_token_level_supervision(
        chosen_content="prefix continue",
        rejected_content="prefix",
        tokenizer=tok,
    )
    rejected_stop_patch = rejected_stop_res["rejected_content"][-1]
    assert rejected_stop_patch["text"] == ""
    assert rejected_stop_patch["tokens"] == []
    assert rejected_stop_patch["ignore_loss"] is False
    assert rejected_stop_patch["rejected_loss"] is True
    g()
    return chosen_stop_res, rejected_stop_res


def test_fork_token_is_last_token(tok):
    last_token_res = compute_token_level_supervision(
        chosen_content="1 2 3",
        rejected_content="1 2 4",
        tokenizer=tok,
    )
    chosen_content = last_token_res["chosen_content"]
    g()
    assert (
        len(chosen_content) == 3
        and chosen_content[-1]["text"] == ""
        and chosen_content[-1]["ignore_loss"]
    ), "The empty post chunk is a non-fork stop token and should ignore its loss"
    return last_token_res


def test_max_chosen_tokens_with_stop(tok):
    tokens_res = compute_token_level_supervision(
        chosen_content="1,2,3,4",
        rejected_content="1,2",
        tokenizer=tok,
        max_chosen_tokens=5,
    )
    g()
    assert tokens_res["chosen_text"] == ",3,4"
    assert tokens_res["chosen_tokens_include_stop"] is True
    return tokens_res


def test_chosen_is_prefix_of_rejected(tok):
    """rejected_text may startswith(chosen_text), which is legal"""
    prefix_res = compute_token_level_supervision(
        chosen_content="Enjoy the sea safely.",
        rejected_content="Enjoy the season now.",
        tokenizer=tok,
    )
    g()
    assert prefix_res["chosen_text"] == " sea"
    assert prefix_res["rejected_text"] == " season"
    assert prefix_res["rejected_text"].startswith(prefix_res["chosen_text"])
    return prefix_res


if __name__ == "__main__":
    from boxx import *
    from test_utils import build_test_tokenizer

    tokenizer = tok = build_test_tokenizer()
    # tokenizer = UnicodeTokenizer()
    first_res, second_res = test_one_token_align_one_patch(tokenizer)
    many_token_align_one_patch_res = test_many_token_align_one_patch(tokenizer)
    many_token_align_many_patch_res = test_many_token_align_many_patch(tokenizer)
    chosen_stop_res, rejected_stop_res = test_fork_token_is_stop_token(tokenizer)
    last_token_res = test_fork_token_is_last_token(tokenizer)
    tokens_res = test_max_chosen_tokens_with_stop(tokenizer)
    prefix_res = test_chosen_is_prefix_of_rejected(tokenizer)
