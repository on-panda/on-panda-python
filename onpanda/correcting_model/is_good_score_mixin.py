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
        messages_clean = [
            {k: msg[k] for k in msg if k not in ["token_level", "correction"]}
            for msg in messages
        ]
        is_good_msgs = self.adapter.build_correction_data_from_token_level(
            messages_clean, is_good=True
        )
        # prefill prmpt_logprobs to get is_good probability
        dic = self.chat_corrector(
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
        first_split_token = list(dic["prompt_logprobs"][-3].values())[0]
        assert (
            first_split_token["decoded_token"] == self.adapter.special_tokens["split"]
        ), (
            "first_split_tokens: "
            f"{dic['prompt_logprobs'][-3]}, "
            "self.adapter.special_tokens['split']: "
            f"{self.adapter.special_tokens['split']}"
        )
        e = 2.718281828459045  # base of the natural logarithm
        prob_first_split = e ** first_split_token["logprob"]
        assert prob_first_split > 0.99, (
            "CorrectingModel should learn to output "
            f"`{self.adapter.special_tokens['split']}` first. "
            f"first_split_tokens: {dic['prompt_logprobs'][-3]}, "
            f"prob_first_split: {prob_first_split}"
        )

        is_good_token = list(dic["prompt_logprobs"][-2].values())[0]
        assert (
            is_good_token["decoded_token"] == self.adapter.special_tokens["is_good"]
        ), (
            "is_good_token: "
            f"{dic['prompt_logprobs'][-2]}, "
            "self.adapter.special_tokens['is_good']: "
            f"{self.adapter.special_tokens['is_good']}"
        )
        is_good_logprob = is_good_token["logprob"]

        second_split_token = list(dic["prompt_logprobs"][-1].values())[0]
        assert (
            second_split_token["decoded_token"] == self.adapter.special_tokens["split"]
        ), (
            "second_split_tokens: "
            f"{dic['prompt_logprobs'][-1]}, "
            "self.adapter.special_tokens['split']: "
            f"{self.adapter.special_tokens['split']}"
        )

        is_good_prob = e**is_good_logprob
        is_good_score = dict(is_good_prob=is_good_prob, is_good_logprob=is_good_logprob)

        if "using chat_correcting.prefill_logprobs for double check" and 0:
            prefill_logprobs = self.chat_corrector.prefill_logprobs(is_good_msgs)[-1][
                "prefill_logprobs"
            ]
            tree - prefill_logprobs
            print(is_good_prob)
            is_good_prob = e ** sum([d["logprob"] for d in prefill_logprobs])
            print(is_good_prob)
            import boxx.g
        return is_good_score

    def compute_best_of_n_score(self, correction_till_goods):
        best_of_n_score = {}
        for till_goods_idx, correction_till_good in enumerate(correction_till_goods):
            for step_idx, correction_step in enumerate(
                correction_till_good["correction_steps"]
            ):
                score = self.compute_is_good_score(
                    correction_step["corrected_messages"]
                )
                best_of_n_score[f"{till_goods_idx}/correction_steps/{step_idx}"] = score
        return best_of_n_score


if __name__ == "__main__":
    from boxx import *
    import transformers
    import mximport

    with mximport.inpkg():
        from .correcting_model import build_test_correcting_model

    correct_model = build_test_correcting_model()

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
