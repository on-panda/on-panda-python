#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:56:53 2025

@author: DIYer22
"""

import os
from copy import deepcopy
import mxlm
import mximport

with mximport.inpkg():
    from .best_of_n_mixin import BestOfNMixin
    from .panda_score_mixin import PandaScoreMixin
    from ..test_utils import build_test_tokenizer
    from ..utils import RESPONSE_ROLES


class CorrectingModel(BestOfNMixin, PandaScoreMixin):
    def __init__(
        self,
        chat_correcting,
        adapter,
        max_correction_attempts=5,
    ):
        self.chat_correcting = chat_correcting
        self.adapter = adapter
        self.max_correction_attempts = max_correction_attempts

    def correct(self, messages):
        correction_prompt = self.adapter.build_correction_prompt(messages)
        correction_response = self.chat_correcting(
            correction_prompt,
            return_dict=True,
            # max_tokens=self.adapter.max_location_tokens + 20,  # bad for reasoning model
        )
        message = correction_response["choices"][0]["message"]
        far_text = message["content"]
        if "new_messages" in correction_response:
            del correction_response["new_messages"]
        apply_res = self.adapter.apply(messages, far_text)
        correction = apply_res["correction"]
        correction["response_info"] = dict(
            model=correction_response.get("model"),
            reasoning=message.get("reasoning", message.get("reasoning_content")),
            finish_reason=correction_response["choices"][0].get("finish_reason"),
        )
        partial_messages = apply_res["partial_messages"]
        return dict(
            correction=correction,
            partial_messages=partial_messages,
            correction_response=correction_response,
        )

    def correct_and_rollout(self, messages, chat_policy, iid_sampling=False):
        """
        Run one correction step.

        If the last message is assistant, retry correction up to max_correction_attempts
        until the correction is not not_found. Failed correction attempts are kept
        in failed_corrections with only correction and correction_response.
        If the final correction is still not_found, regenerate assistant message
        from messages without the last assistant.
        When iid_sampling is True, skip correction and keep the sampled assistant.
        """
        corrected_result = dict(generate_new=False)
        if messages[-1]["role"] in RESPONSE_ROLES:
            failed_corrections = []
            if iid_sampling:
                corrected_result["correction"] = dict(
                    messages_location=dict(
                        not_found=True,
                        is_good=True,
                        match_num=0,
                        find_feedback="iid_sampling: skip correction",
                    )
                )
            else:
                corrected_result["correcting_model_name"] = self.chat_correcting.model
                for _try_idx in range(self.max_correction_attempts):
                    correction_result = self.correct(messages)
                    if not self._is_not_found_correction(
                        correction_result["correction"]
                    ):
                        break
                    failed_corrections.append(
                        dict(
                            correction=correction_result["correction"],
                            correction_response=correction_result[
                                "correction_response"
                            ],
                        )
                    )
                corrected_result["correction"] = correction_result["correction"]
            # corrected_result["correction_response"] = correction_result["correction_response"]
            if failed_corrections:
                corrected_result["failed_corrections"] = failed_corrections
            if self._is_not_found_correction(corrected_result["correction"]):
                # if still not found, regenerate
                rollout_messages = messages[:-1]
                corrected_messages = rollout_messages + [
                    dict(role="assistant", content=chat_policy(rollout_messages))
                ]
                corrected_result["policy_model_name"] = chat_policy.model
                corrected_result["generate_new"] = True
            elif corrected_result["correction"]["messages_location"].get("is_good"):
                corrected_messages = messages
            else:
                partial_messages = correction_result["partial_messages"]
                if partial_messages[-1].get("finish_reason") == "stop":
                    corrected_messages = partial_messages
                else:
                    corrected_content = chat_policy(
                        partial_messages,
                        continue_final_message=True,
                        add_generation_prompt=False,
                        echo=True,
                    )
                    corrected_result["policy_model_name"] = chat_policy.model
                    corrected_messages = messages[:-1] + [
                        dict(role="assistant", content=corrected_content)
                    ]
        else:  # make new message
            # Corrected result without correcting_model_name key means new message.
            corrected_messages = messages + [
                dict(role="assistant", content=chat_policy(messages))
            ]
            corrected_result["policy_model_name"] = chat_policy.model
            corrected_result["generate_new"] = True

        corrected_result["corrected_messages"] = corrected_messages[:]
        # g()
        return corrected_result

    def iterative_correction_till_good(
        self, messages, chat_policy, max_rollouts=5, iid_sampling=False
    ):
        """
        Run iterative correction for max_rollouts steps.

        Step count matches rollout count. If input already ends with assistant,
        that preset response consumes one rollout step.

        Correction data is attached to the previous step it evaluates.
        When correction is_good, no new step is appended.
        """
        applied_corrections = 0
        correction_steps = []

        if max_rollouts > 0 and messages[-1]["role"] in RESPONSE_ROLES:
            correction_steps.append(
                dict(
                    generate_new=False,
                    corrected_messages=messages[:],
                    applied_corrections=applied_corrections,
                )
            )

        while len(correction_steps) < max_rollouts:
            if not correction_steps:
                first_result = self.correct_and_rollout(
                    messages, chat_policy, iid_sampling=iid_sampling
                )
                first_result["applied_corrections"] = applied_corrections
                correction_steps.append(first_result.copy())
                continue

            current_messages = correction_steps[-1]["corrected_messages"]
            corrected_result = self.correct_and_rollout(
                current_messages, chat_policy, iid_sampling=iid_sampling
            )
            correction_steps[-1].update(
                {
                    k: corrected_result.pop(k)
                    for k in (
                        "correction",
                        "correcting_model_name",
                        "failed_corrections",
                    )
                    if k in corrected_result
                }
            )
            correction = correction_steps[-1].get("correction")
            if correction is not None:
                if corrected_result["generate_new"]:
                    applied_corrections = 0
                elif self._is_applied_correction(correction):
                    applied_corrections += 1
                if correction["messages_location"].get("is_good"):
                    break

            corrected_result["applied_corrections"] = applied_corrections
            correction_steps.append(corrected_result.copy())

        result = correction_steps[-1].copy()
        result["max_rollouts"] = max_rollouts
        result["correction_steps"] = correction_steps
        return result

    def iterative_correction(
        self,
        messages,
        chat_policy,
        rollout_num=5,
        mode="till_good",
        iid_sampling=False,
    ):
        """
        Run iterative correction mode in one of:
        - till_good: one iterative_correction_till_good with max_rollouts=rollout_num.
        - best_of_n/pass_at_k: repeatedly call iterative_correction_till_good
          from the same start messages until total consumed steps reaches rollout_num.
        """
        correction_till_goods = []
        if mode not in (
            "till_good",
            "best_of_n",
            "pass_at_k",
        ):
            raise ValueError(
                f"Unknown iterative_correction mode: {mode}, "
                "expected one of [till_good, best_of_n, pass_at_k]"
            )
        remaining_rollouts = rollout_num
        while remaining_rollouts > 0:
            till_good_messages = messages
            if (
                correction_till_goods and messages[-1]["role"] in RESPONSE_ROLES
            ):  # if already has assistant response, remove it for second round correction
                till_good_messages = mxlm.remove_last_assistant(messages)
            correction_till_good = self.iterative_correction_till_good(
                till_good_messages,
                chat_policy,
                max_rollouts=remaining_rollouts,
                iid_sampling=iid_sampling,
            )
            correction_till_goods.append(correction_till_good)
            remaining_rollouts -= len(correction_till_good["correction_steps"])
            if mode == "till_good":
                break

        aggregated_result = dict(
            iterative_correction_mode=mode,
            rollout_num=rollout_num,
            adapter_info=getattr(self.adapter, "info", None),
        )
        if mode == "till_good":
            chosen_correction_step = correction_till_goods[0]["correction_steps"][-1]
            aggregated_result.update(chosen_correction_step)
        elif mode == "best_of_n":
            delta_result = self.choose_best_of_n(correction_till_goods)
            aggregated_result.update(delta_result)
        aggregated_result["correction_till_goods"] = correction_till_goods
        return aggregated_result

    def _is_not_found_correction(self, correction):
        messages_location = correction["messages_location"]
        return not messages_location.get("is_good") and messages_location.get(
            "not_found"
        )

    def _is_applied_correction(self, correction):
        return bool(correction["messages_location"].get("path_keys"))


def build_test_correcting_model(
    chat_correcting=None,
    adapter=None,
    tokenizer="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
):
    import onpanda

    if chat_correcting is None:
        chat_correcting = mxlm.ChatAPI(
            model="peqwen3-sft-cm-it1000",
            temperature=0,
            top_p=1.0,
            max_tokens=40,
            logprobs=True,
            return_dict=True,
            is_reasoning=False,
        )
    if isinstance(tokenizer, str):
        tokenizer = build_test_tokenizer(tokenizer)
    if adapter is None:
        adapter = onpanda.FindAndReplaceCorrectionAdapter(
            tokenizer=tokenizer,
            special_tokens=dict(
                split="<|fim_pad|>",  # for qwen 2.5
                stop="<|fim_suffix|>",
                is_good="<|fim_prefix|>",
                reasoning="<|fim_middle|>",
            ),
        )
    return CorrectingModel(
        chat_correcting,
        adapter,
    )


def build_reasoning_correcting_model(
    chat_correcting=None,
    adapter=None,
    tokenizer="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
):
    from dotenv import load_dotenv
    from pathlib import Path
    import os
    import onpanda

    load_dotenv(Path(__file__).resolve().parents[2] / ".env.local")
    if chat_correcting is None:
        chat_correcting = mxlm.ChatAPI(
            base_url=os.environ.get(
                "REASONING_CM_BASE_URL", "https://api.inference.wandb.ai/v1"
            ),
            api_key=os.environ.get(
                "REASONING_CM_API_KEY", os.environ.get("WANDB_API_KEY")
            ),
            model=os.environ.get("REASONING_CM_MODEL", "moonshotai/Kimi-K2.6"),
            temperature=1,
            top_p=0.95,
            max_tokens=1024 * 10,
            is_reasoning=True,
        )
    if adapter is None:
        adapter = onpanda.FindAndReplaceCorrectionAdapter(
            tokenizer=tokenizer,
        )
    return CorrectingModel(
        chat_correcting,
        adapter,
    )


def build_correcting_model_with_policy(reasoning=True):
    new_kwargs = dict(max_tokens=1536, temperature=0.8)
    if reasoning:
        correcting_model = build_reasoning_correcting_model()
        new_kwargs["model"] = os.environ.get(
            "POLICY_API_MODEL", "Qwen/Qwen3.6-35B-A3B"
        )  # should support continue_final_message
    else:
        correcting_model = build_test_correcting_model()
    chat_policy = deepcopy(correcting_model.chat_correcting)
    extra_parameters_json5_str = os.environ.get("POLICY_API_EXTRA_PARAMETERS_JSON5", "")
    if extra_parameters_json5_str:
        import json5

        extra_parameters = json5.loads(extra_parameters_json5_str)
        new_kwargs.update(extra_parameters)
    chat_policy.default_kwargs.update(new_kwargs)
    return dict(correcting_model=correcting_model, chat_policy=chat_policy)


if __name__ == "__main__":
    from boxx import *
    from onpanda.test_utils import get_test_rejected_msgs1

    _d = build_correcting_model_with_policy()
    correcting_model, chat_policy = _d["correcting_model"], _d["chat_policy"]

    msgs = [
        {"role": "user", "content": "5+7="},
        {"role": "assistant", "content": "32"},
        # {"role": "assistant", "content": "12"},
    ]
    msgs = get_test_rejected_msgs1()[0]

    # msgs = [{"role": "user", "content": "How many `1` in result of 652*8596"},]

    corrected = correcting_model.iterative_correction(
        msgs,
        chat_policy,
        rollout_num=3,
        mode="best_of_n",
        # iid_sampling=True,
    )
    tree(corrected)
