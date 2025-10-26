#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:56:53 2025

@author: DIYer22
"""


class IsGoodScoreMixin:
    def compute_is_good_score(
        self,
        messages,
    ):
        # add correcting SFT system prompt and is_good answer (`<|split|><|split|>`)
        is_good_msgs = self.builder.build_correcting_sft_by_token_level_SFT(
            messages, is_good=True
        )
        # prefill prmpt_logprobs to get is_good probability
        dic = self.chat_correcting(
            is_good_msgs,
            return_dict=True,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            logprobs=True,
            # top_logprobs=1,
            extra_body=dict(
                prompt_logprobs=True,
                add_generation_prompt=False,
                continue_final_message=True,
                skip_special_tokens=False,
            ),
        )
        first_split_token = list(dic["prompt_logprobs"][-2].values())[0]
        assert (
            first_split_token["decoded_token"] == self.builder.SPLIT_TOKEN
        ), f"first_split_tokens: {dic['prompt_logprobs'][-2]}, self.builder.SPLIT_TOKEN: {self.builder.SPLIT_TOKEN}"
        e = 2.718281828459045  # base of the natural logarithm
        prob_first_split = e ** first_split_token["logprob"]
        assert (
            prob_first_split > 0.99
        ), f"CorrectingSftModel should learn to output `{self.builder.SPLIT_TOKEN}` first. first_split_tokens: {dic['prompt_logprobs'][-2]}, prob_first_split: {prob_first_split}"

        second_split_token = list(dic["prompt_logprobs"][-1].values())[0]
        assert (
            second_split_token["decoded_token"] == self.builder.SPLIT_TOKEN
        ), f"second_split_tokens: {dic['prompt_logprobs'][-1]}, self.builder.SPLIT_TOKEN: {self.builder.SPLIT_TOKEN}"
        is_good_logprob = second_split_token["logprob"]
        is_good_prob = e**is_good_logprob
        is_good_score = dict(is_good_prob=is_good_prob, is_good_logprob=is_good_logprob)

        if "using chat_correcting.prefill_logprobs for double check" and 0:
            prefill_logprobs = self.chat_correcting.prefill_logprobs(is_good_msgs)[-1][
                "prefill_logprobs"
            ]
            tree - prefill_logprobs
            print(is_good_prob)
            is_good_prob = e ** sum([d["logprob"] for d in prefill_logprobs])
            print(is_good_prob)
            g()
        return is_good_score


if __name__ == "__main__":
    from boxx import *
    import transformers
    import mximport

    with mximport.inpkg():
        from .correcting_sft_model import build_test_correcting_sft_model

    correct_model = build_test_correcting_sft_model()

    msgs = [
        {"role": "user", "content": "5+7="},
        {"role": "assistant", "content": "32"},
        # {"role": "assistant", "content": "12"},
    ]
    msgs = [
        {"role": "user", "content": "Name three kinds of fruit:"},
        {
            "role": "assistant",
            "content": "Apple, potato, banana.",
            # "content": "Apple, orange, banana.",
        },
    ]

    is_good_score = correct_model.compute_is_good_score(msgs)
    print(f'{is_good_score["is_good_prob"]*100:04.1f}%')
