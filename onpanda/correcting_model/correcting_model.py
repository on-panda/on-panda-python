#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:56:53 2025

@author: DIYer22
"""
import mximport

with mximport.inpkg():
    from .is_good_score_mixin import IsGoodScoreMixin
    from .panda_score_mixin import PandaScoreMixin
    from ..test_utils import build_test_tokenizer
    from ..utils import RESPONSE_ROLES


class CorrectingModel(IsGoodScoreMixin, PandaScoreMixin):
    def __init__(
        self,
        chat_corrector,
        adapter,
        max_correction_attempts=5,
    ):
        self.chat_corrector = chat_corrector
        self.adapter = adapter
        self.max_correction_attempts = max_correction_attempts

    def correct(self, messages):
        correction_prompt = self.adapter.build_correction_prompt(messages)
        corrector_response = self.chat_corrector(
            correction_prompt,
            return_dict=True,
            # max_tokens=self.adapter.max_location_tokens + 20,  # bad for reasoning model
        )
        far_text = corrector_response["choices"][0]["message"]["content"]
        if "new_messages" in corrector_response:
            del corrector_response["new_messages"]
        apply_res = self.adapter.apply(messages, far_text)
        correction = apply_res["correction"]
        partial_messages = apply_res["partial_messages"]
        return dict(
            correction=correction,
            partial_messages=partial_messages,
            corrector_response=corrector_response,
        )

    def correct_and_rollout(self, messages, chat_policy):
        """
        Run one correction step.

        If the last message is assistant, retry correction up to max_correction_attempts
        until the correction is not not_found. Failed correction attempts are kept
        in failed_corrections with only correction and corrector_response.
        If the final correction is still not_found, regenerate assistant message
        from messages without the last assistant.
        """
        corrected_result = dict(policy_model=chat_policy.model, generate_new=False)
        if messages[-1]["role"] in RESPONSE_ROLES:
            corrected_result["corrector_model"] = self.chat_corrector.model
            failed_corrections = []
            for _try_idx in range(self.max_correction_attempts):
                correction_result = self.correct(messages)
                if not self._is_not_found_correction(correction_result["correction"]):
                    break
                failed_corrections.append(
                    dict(
                        correction=correction_result["correction"],
                        corrector_response=correction_result["corrector_response"],
                    )
                )
            corrected_result["correction"] = correction_result["correction"]
            if failed_corrections:
                corrected_result["failed_corrections"] = failed_corrections
            if self._is_not_found_correction(corrected_result["correction"]):
                # if still not found, regenerate
                rollout_messages = messages[:-1]
                corrected_messages = rollout_messages + [
                    dict(role="assistant", content=chat_policy(rollout_messages))
                ]
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
                    corrected_messages = messages[:-1] + [
                        dict(role="assistant", content=corrected_content)
                    ]
        else:  # make new message
            # Corrected result without corrector_model key means new message.
            corrected_messages = messages + [
                dict(role="assistant", content=chat_policy(messages))
            ]
            corrected_result["generate_new"] = True

        corrected_result["corrected_messages"] = corrected_messages[:]
        # g()
        return corrected_result

    def iterative_correction(self, messages, chat_policy, max_rollouts=5):
        """
        Run iterative correction for max_rollouts steps.

        If a step returns correction.not_found, reset applied_corrections to zero
        and continue the next rollout.
        """
        applied_corrections = 0
        corrected_messages = messages
        correction_steps = []
        for _rollout_idx in range(max_rollouts):
            corrected_result = self.correct_and_rollout(corrected_messages, chat_policy)
            corrected_messages = corrected_result["corrected_messages"]
            correction = corrected_result.get("correction")
            if corrected_result["generate_new"]:
                applied_corrections = 0
            elif self._is_applied_correction(correction):
                applied_corrections += 1
            corrected_result["applied_corrections"] = applied_corrections
            correction_steps.append(corrected_result.copy())
            if correction and correction["messages_location"].get("is_good"):
                break
        corrected_result["applied_corrections"] = applied_corrections
        corrected_result["max_rollouts"] = max_rollouts
        corrected_result["correction_steps"] = correction_steps
        return corrected_result

    def _is_not_found_correction(self, correction):
        messages_location = correction["messages_location"]
        return not messages_location.get("is_good") and messages_location.get(
            "not_found"
        )

    def _is_applied_correction(self, correction):
        return bool(correction["messages_location"].get("path_keys"))


def build_test_correcting_model(
    chat_corrector=None,
    adapter=None,
    tokenizer="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
):
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
        chat_corrector,
        adapter,
    )


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
    corrected = correcting_model.iterative_correction(msgs, chat_policy, max_rollouts=5)
    tree(corrected)
