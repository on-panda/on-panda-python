from typing import List, Dict, Tuple

STOP_TOKEN_DEFAULT = "<|stop|>"  # placement for stop token


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
           tokenizer.decode(tokens[start:end], skip_special_tokens=True),
           add_special_tokens=False
      )``

    This is the formal definition of a *patch* in the docstring.
    """
    if isinstance(tokens, str):
        tokens = tokenizer.encode(tokens, add_special_tokens=False)
    n = len(tokens)
    # First try the trivial 1‑token span; for most Latin text it succeeds.
    start, end = center_idx, center_idx + 1

    def reversible(s: int, e: int) -> bool:
        sub = tokens[s:e]
        text = tokenizer.decode(sub, skip_special_tokens=True)
        return tokenizer.encode(text, add_special_tokens=False) == sub

    if reversible(start, end):
        return start, end
    for token_num in range(1, n + 1):
        for bias in range(token_num):
            start = center_idx - bias
            end = start + token_num
            if start >= 0 and end <= n:
                if reversible(start, end):
                    return start, end

    # Fallback: the whole sequence (should never happen in normal text).
    return 0, n


def compute_token_level_supervision(
    *,
    rejected_content: str,
    chosen_content: str,
    tokenizer=None,
    STOP_TOKEN=STOP_TOKEN_DEFAULT,
) -> Dict:
    """
    Compute token‑level supervision signals.

    Parameters
    ----------
    chosen_content / rejected_content : str
        The two candidate strings – *chosen* is forked from *rejected*.
    tokenizer : PreTrainedTokenizerBase, optional

    Returns
    -------
    dict  with keys

        fork_token_idx : int
            Index of the first diverging token position.
        chosen_token_id / rejected_token_id : int
            Token‑IDs at that fork position.
        chosen_content / rejected_content : list[dict]
            Each list is ``[pre_patch, patch, post_patch]`` where every
            element is a dict ``{"tokens": List[int], "ignore_loss": bool}``.
            • For *chosen*, the patch always receives loss (``ignore_loss=False``).
            • For *rejected*, the patch receives rejected loss **only if**
              it is a *single* token.  When the patch spans > 1 token we set ``ignore_loss=True`` for
              safety, because a many‑token rejected is harmful.
    """
    from transformers import AutoTokenizer

    if tokenizer is None:
        tokenizer = UnicodeTokenizer()

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

    chosen_token_id = tok_ch[fork_idx] if fork_idx < len(tok_ch) else STOP_TOKEN
    rejected_token_id = tok_rj[fork_idx] if fork_idx < len(tok_rj) else STOP_TOKEN

    # 2️⃣ Locate minimal reversible patches around the diverging tokens.
    ch_s, ch_e = _minimal_reversible_patch(tok_ch, fork_idx, tokenizer)
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
            "text": tokenizer.decode(tokens[:s], skip_special_tokens=True),
            "ignore_loss": True,
        }
        patch = {
            "type": "text",
            "text": tokenizer.decode(tokens[s:e], skip_special_tokens=True),
            "ignore_loss": ignore_patch_loss,
            "tokens": tokens[s:e],
        }
        if rejected_loss:
            patch["rejected_loss"] = True
        post = {
            "type": "text",
            "text": tokenizer.decode(tokens[e:], skip_special_tokens=True),
            "ignore_loss": True,
        }
        if is_fork_on_stop:
            return [pre, patch]
        return [pre, patch, post]

    chosen_chunks = build_chunks(
        tok_ch, ch_s, ch_e, is_fork_on_stop=chosen_token_id == STOP_TOKEN
    )
    # Only apply negative loss when the rejected patch is exactly one token.
    rejected_chunks = build_chunks(
        tok_rj,
        rj_s,
        rj_e,
        ignore_patch_loss=(rj_e - rj_s != 1),
        rejected_loss=True,
        is_fork_on_stop=rejected_token_id == STOP_TOKEN,
    )
    # set chosen_text and rejected_text
    chosen_text = next(
        chunk for chunk in chosen_chunks if not chunk.get("ignore_loss")
    )["text"]
    rejected_text = next(
        chunk for chunk in rejected_chunks if chunk.get("rejected_loss")
    )["text"]
    return {
        "fork_token_idx": fork_idx,
        "chosen_token_id": chosen_token_id,
        "rejected_token_id": rejected_token_id,
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
        self.name_or_path = "UnicodeTokenizer"

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
    res = compute_token_level_supervision(
        chosen_content="I love cats.", rejected_content="I love dogs.", tokenizer=tok
    )
    assert _patch_len(res, "chosen", tok) == 1
    assert _patch_len(res, "rejected", tok) == 1
    # Negative loss **is** applied to the rejected patch.

    assert res["rejected_content"][1]["ignore_loss"] is False
    g()

    res = compute_token_level_supervision(
        chosen_content="Answer is中国",
        rejected_content="Answer is Chinese",
        tokenizer=tok,
    )
    g()
    # assert len(res["chosen_content"][1]["text"]) > 1


def test_many_token_align_one_patch(tok):
    """Chosen patch >1 token, rejected patch 1 token."""
    res = compute_token_level_supervision(
        chosen_content="I like '🥢'", rejected_content="I like '🥄'", tokenizer=tok
    )
    g()
    assert _patch_len(res, "chosen", tok) > 1
    assert _patch_len(res, "rejected", tok) > 1
    # Negative loss still applies (rejected patch single token).
    assert res["rejected_content"][1]["ignore_loss"] is True


def test_many_token_align_many_patch(tok):
    """' 🥢' == [11162, 98, 95]"""
    res = compute_token_level_supervision(
        chosen_content="prefix 🥢subfix",  # with space
        rejected_content="prefix🥢subfix",  # without space
        tokenizer=tok,
    )
    rejected_patch = res["rejected_content"][1]
    g()
    if len(rejected_patch["tokens"]) > 1:
        assert rejected_patch[
            "ignore_loss"
        ], "if rejected_patch include many tokens, ignore loss. Because a many‑token rejected loss is harmful."


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
    g()


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
    ), "The stop token is last patch with empty string. Should ignore it's loss"


if __name__ == "__main__":
    from boxx import *
    from test_utils import build_test_tokenizer

    tokenizer = tok = build_test_tokenizer()
    # tokenizer = UnicodeTokenizer()
    test_one_token_align_one_patch(tokenizer)
    # test_many_token_align_one_patch(tokenizer)
    # test_one_to_many(tokenizer)
    test_many_token_align_many_patch(tokenizer)
    test_fork_token_is_stop_token(tokenizer)
    test_fork_token_is_last_token(tokenizer)
