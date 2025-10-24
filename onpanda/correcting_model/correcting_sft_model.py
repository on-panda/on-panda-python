#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:56:53 2025

@author: DIYer22
"""
import mximport

with mximport.inpkg():
    from .is_good_score_mixin import IsGoodScoreMixin


class TokenLevelCorrectingModelMeta:
    """
    Meta class of different token-level correcting models. e.g.:
    - CorrectingCopyResponseModel: Using Copy response to get whole context for each generated token, and better computation for each token
    - CorrectingSftModel: Next Token Prediction as correcting location
    - Reasoning for Correcting using JSON output with right prefix
    - Bidirectional correcting head for location with whole context
    """

    def __init__(self):
        pass

    def correct(self, text: str) -> str:
        # Implement token-level correction logic here
        return text


class CorrectingSftModel(TokenLevelCorrectingModelMeta, IsGoodScoreMixin):
    def __init__(
        self,
        chat,
        sft_correcting_builder,
    ):
        self.chat = chat
        self.builder = sft_correcting_builder


def build_test_correcting_sft_model(chat=None, builder=None):
    import mxlm
    import onpanda
    import transformers

    if chat is None:
        chat = mxlm.ChatAPI(model="step1f-correct-sft-it1200")
    if builder is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            use_fast=True,
            local_files_only=True,
        )
        builder = onpanda.NextTokenPredictionAsCorrectingBuilder(
            tokenizer=tokenizer,
            SPLIT_TOKEN="<|fim_pad|>",  # for qwen 2.5
            STOP_TOKEN="<|fim_suffix|>",
        )
    return CorrectingSftModel(chat, builder)


if __name__ == "__main__":
    pass
