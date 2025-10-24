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
    TODO: using this or delete
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

    def build_correcting_prompt(self, msgs):

        sys_prompt_message = dict(
            role="system",
            content=self.builder.get_correcting_sft_system_prompt(),
        )
        return msgs + [sys_prompt_message]

    def correct(self, msgs):
        """
        Input QA msgs, return unicode_location
        """
        correcting_prompt = self.build_correcting_prompt(msgs)
        response_dic = self.chat(
            correcting_prompt,
            return_dict=True,
            max_tokens=self.builder.max_location_tokens + 20,
        )
        ntp_as_correcting_text = response_dic["choices"][0]["message"]["content"]
        corrected = self.builder.apply_ntp_as_correcting(msgs, ntp_as_correcting_text)
        return corrected


def build_test_correcting_sft_model(chat=None, builder=None):
    import mxlm
    import onpanda
    import transformers

    if chat is None:
        chat = mxlm.ChatAPI(
            model="step1f-correct-sft-it1200",
            temperature=0,
            logprobs=True,
            return_dict=True,
            max_tokens=40,
        )
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
    from boxx import *
    from onpanda.test_utils import get_test_rejected_msgs1

    correct_model = build_test_correcting_sft_model()

    msgs = [
        {"role": "user", "content": "5+7="},
        {"role": "assistant", "content": "32"},
        # {"role": "assistant", "content": "12"},
    ]
    msgs = get_test_rejected_msgs1()[0]

    corrected = correct_model.correct_sample(msgs)
    tree(corrected)
