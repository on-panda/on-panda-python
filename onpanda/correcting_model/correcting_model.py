#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:56:53 2025

@author: DIYer22
"""
import mximport

with mximport.inpkg():
    from .is_good_score_mixin import IsGoodScoreMixin


class CorrectingModelBase:
    """
    TODO: using this or delete
    Base class of different token-level correcting models. e.g.:
    - CorrectingCopyResponseModel: Using Copy response to get whole context for each generated token, and better computation for each token
    - CorrectingModel: Find-and-replace correction on token-level supervision
    - Reasoning for Correcting using JSON output with right prefix
    - Bidirectional correcting head for location with whole context
    """

    def __init__(self):
        pass

    def correct(self, text: str) -> str:
        # Implement token-level correction logic here
        return text


class CorrectingModel(CorrectingModelBase, IsGoodScoreMixin):
    def __init__(
        self,
        chat_corrector,
        adapter,
    ):
        self.chat_corrector = chat_corrector
        self.adapter = adapter

    def correcting(self, messages):
        correction_prompt = self.adapter.build_correction_prompt(messages)
        response_dic = self.chat_corrector(
            correction_prompt,
            return_dict=True,
            max_tokens=self.adapter.max_location_tokens + 20,
        )
        far_text = response_dic["choices"][0]["message"]["content"]
        apply_res = self.adapter.apply(messages, far_text)
        correction = apply_res["correction"]
        is_good = correction["find_and_replace"].get("is_good")
        partial_messages = apply_res["partial_messages"]
        return dict(
            correction=correction,
            partial_messages=partial_messages,
            is_good=is_good,
        )

    def generate_and_apply_correction(self, messages, chat_policy):
        corrected = dict(model_policy=chat_policy.model)
        if messages[-1]["role"] in ["assistant"]:
            corrected["model_correcting"] = self.chat_corrector.model
            correcting_result = self.correcting(messages)
            corrected["correction"] = correcting_result["correction"]
            if correcting_result["is_good"]:
                corrected_messages = messages
            else:
                partial_messages = correcting_result["partial_messages"]
                if partial_messages[-1].get("finish_reason") == "stop":
                    corrected_messages = partial_messages
                else:
                    corrected_content = chat_policy(
                        partial_messages,
                        continue_final_message=True,
                        add_generation_prompt=False,
                        echo=True,
                    )
                    corrected_messages = messages[:-1] + [
                        dict(role="assistant", content=corrected_content)
                    ]
        else:  # make new message
            # corrected without model_correcting key means new message
            corrected_messages = messages + [
                dict(role="assistant", content=chat_policy(messages))
            ]

        corrected["corrected_messages"] = corrected_messages
        # g()
        return corrected

    def iterative_correction(self, messages, chat_policy, max_corrections=5):
        correction_count = 0
        corrected_messages = messages
        correcteds = []
        for _correction_idx in range(max_corrections):
            corrected = self.generate_and_apply_correction(
                corrected_messages, chat_policy
            )
            correcteds.append(corrected.copy())
            corrected_messages = corrected["corrected_messages"]
            if "correction" in corrected and corrected["correction"][
                "find_and_replace"
            ].get("is_good"):
                break
            correction_count += 1
        corrected["correction_count"] = correction_count
        corrected["max_corrections"] = max_corrections
        corrected["correcteds"] = correcteds
        return corrected


def build_test_correcting_model(chat_corrector=None, adapter=None):
    import mxlm
    import onpanda
    import transformers

    if chat_corrector is None:
        chat_corrector = mxlm.ChatAPI(
            model="step1f-correct-sft-it1200",
            temperature=0,
            top_p=1.0,
            max_tokens=40,
            logprobs=True,
            return_dict=True,
        )
    if adapter is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            use_fast=True,
            # local_files_only=True,
        )
        adapter = onpanda.FindAndReplaceCorrectionAdapter(
            tokenizer=tokenizer,
            special_tokens=dict(
                split="<|fim_pad|>",  # for qwen 2.5
                stop="<|fim_suffix|>",
                is_good="<|fim_prefix|>",
                reasoning="<|fim_middle|>",
            ),
        )
    return CorrectingModel(chat_corrector, adapter)


if __name__ == "__main__":
    from boxx import *
    from onpanda.test_utils import get_test_rejected_msgs1
    from copy import deepcopy

    correcting_model = build_test_correcting_model()

    msgs = [
        {"role": "user", "content": "5+7="},
        {"role": "assistant", "content": "32"},
        # {"role": "assistant", "content": "12"},
    ]
    msgs = get_test_rejected_msgs1()[0]

    # msgs = [{"role": "user", "content": "How many `1` in result of 652*8596"},]

    chat_policy = deepcopy(correcting_model.chat_corrector)
    chat_policy.default_kwargs["max_tokens"] = 1536
    corrected = correcting_model.iterative_correction(
        msgs, chat_policy, max_corrections=5
    )
    tree(corrected)
