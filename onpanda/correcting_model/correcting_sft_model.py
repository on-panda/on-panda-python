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
        chat_correcting,
        sft_correcting_builder,
    ):
        self.chat_correcting = chat_correcting
        self.builder = sft_correcting_builder

    def build_correcting_prompt(self, msgs):

        sys_prompt_message = dict(
            role="system",
            content=self.builder.get_correcting_sft_system_prompt(),
        )
        return msgs + [sys_prompt_message]

    def generate_correction(self, msgs):
        """
        Input QA msgs, return unicode_location
        """
        correcting_prompt = self.build_correcting_prompt(msgs)
        response_dic = self.chat_correcting(
            correcting_prompt,
            return_dict=True,
            max_tokens=self.builder.max_location_tokens + 20,
        )
        ntp_as_correcting_text = response_dic["choices"][0]["message"]["content"]
        correction = self.builder.apply_ntp_as_correcting(msgs, ntp_as_correcting_text)
        return correction

    def generate_and_apply_correction(self, msgs, chat_policy):
        corrected = dict(model_policy=chat_policy.model)
        if msgs[-1]["role"] in ["assistant"]:
            corrected["model_correcting"] = self.chat_correcting.model
            correction = self.generate_correction(msgs)
            corrected["correction"] = correction
            if correction["ntp_as_correcting"].get("is_good"):
                corrected_msgs = msgs
            else:
                partial_messages = correction["partial_messages"]
                if partial_messages[-1].get("finish_reason") == "stop":
                    corrected_msgs = partial_messages
                else:
                    corrected_content = chat_policy(
                        partial_messages,
                        continue_final_message=True,
                        add_generation_prompt=False,
                        echo=True,
                    )
                    corrected_msgs = msgs[:-1] + [
                        dict(role="assistant", content=corrected_content)
                    ]
        else:  # make new message
            # corrected without model_correcting key means new message
            corrected_msgs = msgs + [dict(role="assistant", content=chat_policy(msgs))]

        corrected["corrected_messages"] = corrected_msgs
        # g()
        return corrected

    def correcting_sampling(self, msgs, chat_policy, n=5):
        correction_count = 0
        corrected_msgs = msgs
        correcteds = []
        for correction_idx in range(n):
            corrected = self.generate_and_apply_correction(corrected_msgs, chat_policy)
            correcteds.append(corrected.copy())
            corrected_msgs = corrected["corrected_messages"]
            if "correction" in corrected and corrected["correction"][
                "ntp_as_correcting"
            ].get("is_good"):
                break
            correction_count += 1
        corrected["correction_count"] = correction_count
        corrected["correction_count_max"] = n
        corrected["correcteds"] = correcteds
        return corrected


def build_test_correcting_sft_model(chat_correcting=None, builder=None):
    import mxlm
    import onpanda
    import transformers

    if chat_correcting is None:
        chat_correcting = mxlm.ChatAPI(
            model="step1f-correct-sft-it1200",
            temperature=0,
            top_p=1.0,
            max_tokens=40,
            logprobs=True,
            return_dict=True,
        )
    if builder is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            use_fast=True,
            # local_files_only=True,
        )
        builder = onpanda.NextTokenPredictionAsCorrectingBuilder(
            tokenizer=tokenizer,
            SPLIT_TOKEN="<|fim_pad|>",  # for qwen 2.5
            STOP_TOKEN="<|fim_suffix|>",
        )
    return CorrectingSftModel(chat_correcting, builder)


if __name__ == "__main__":
    from boxx import *
    from onpanda.test_utils import get_test_rejected_msgs1
    from copy import deepcopy

    correct_model = build_test_correcting_sft_model()

    msgs = [
        {"role": "user", "content": "5+7="},
        {"role": "assistant", "content": "32"},
        # {"role": "assistant", "content": "12"},
    ]
    msgs = get_test_rejected_msgs1()[0]

    # msgs = [{"role": "user", "content": "How many `1` in result of 652*8596"},]

    chat_policy = deepcopy(correct_model.chat_correcting)
    chat_policy.default_kwargs["max_tokens"] = 1536
    corrected = correct_model.correcting_sampling(msgs, chat_policy, n=5)
    tree(corrected)
