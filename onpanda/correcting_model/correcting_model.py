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
    from ..test_utils import (
        ERROR_TYPES,
        build_test_tokenizer,
        get_test_trajectories,
    )
    from ..utils import RESPONSE_ROLES


def take_policy_message(policy_response):
    """`reasoning_content` is the same channel as `reasoning`, which response templates read."""
    choice = policy_response["choices"][0]
    message = choice["message"]
    reasoning_content = message.pop("reasoning_content", None)
    if reasoning_content:
        message["reasoning"] = reasoning_content
    elif not message.get("reasoning"):
        message.pop("reasoning", None)
    if not message.get("tool_calls"):
        message.pop("tool_calls", None)
    if choice.get("finish_reason") is not None:
        message["finish_reason"] = choice["finish_reason"]
        if (
            message["finish_reason"] == "stop"
            and message.get("tool_calls")
            and message["tool_calls"] != [{}]
        ):
            message["finish_reason"] = "tool_calls"
    return message


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

    def correct(self, messages, tools=None):
        correction_prompt = self.adapter.build_correction_prompt(messages)
        correction_response = self.chat_correcting(
            correction_prompt,
            return_dict=True,
            skip_special_tokens=False,
            # max_tokens=self.adapter.max_location_tokens + 20,  # bad for reasoning model
            # The correcting model answers in the FAR text format, it never calls a tool itself.
            **(dict(tools=tools, tool_choice="none") if tools else {}),
        )
        message = correction_response["choices"][0]["message"]
        far_text = message["content"]
        if "new_messages" in correction_response:
            del correction_response["new_messages"]
        apply_res = self.adapter.apply(messages, far_text, tools=tools)
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

    def correct_and_rollout(
        self,
        messages,
        chat_policy,
        iid_sampling=False,
        adapter_policy=None,
        tools=None,
    ):
        """
        Run one correction step.

        If the last message is assistant, retry correction up to max_correction_attempts
        until the correction is not not_found. Failed correction attempts are kept
        in failed_corrections with only correction and correction_response.
        If the final correction is still not_found, regenerate assistant message
        from messages without the last assistant.
        When iid_sampling is True, skip correction and keep the sampled assistant.

        adapter_policy is the policy model's adapter, defaults to the correcting one. Its tokenizer
        and response_template render and align the continuation prefix, then parse it back.
        """
        adapter_policy = adapter_policy or self.adapter
        policy_kwargs = dict(skip_special_tokens=False)
        if tools:
            policy_kwargs["tools"] = tools
        # A continuation comes back as raw text, so the server must not parse tool calls out of it.
        continuation_kwargs = policy_kwargs.copy()
        if tools:
            continuation_kwargs["tool_choice"] = "none"
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
                    correction_result = self.correct(messages, tools=tools)
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
                    take_policy_message(
                        chat_policy(rollout_messages, return_dict=True, **policy_kwargs)
                    )
                ]
                corrected_result["policy_model_name"] = chat_policy.model
                corrected_result["generate_new"] = True
            elif corrected_result["correction"]["messages_location"].get("is_good"):
                corrected_messages = messages
            else:  # has correction
                partial_messages = correction_result["partial_messages"]
                corrected_message_index = len(partial_messages) - 1
                partial_templated = adapter_policy.build_partial_templated_prompt(
                    messages[corrected_message_index], partial_messages[-1]
                )
                prefix = partial_templated["templated_prompt"]
                corrected_result["correction"]["continue_prefix_right40"] = (
                    prefix if len(prefix) <= 40 else "..." + prefix[-37:]
                )
                complete_templated_prompt = adapter_policy.response_template.apply(
                    partial_messages[-1]
                )["templated_prompt"]
                if partial_messages[-1].get("finish_reason") in (
                    "stop",
                    "tool_calls",
                ) and adapter_policy.tokenizer.encode(
                    prefix, add_special_tokens=False
                ) == adapter_policy.tokenizer.encode(
                    complete_templated_prompt, add_special_tokens=False
                ):  # correction to stop
                    corrected_messages = partial_messages
                else:  # continue_final_message with policy response_template
                    policy_choice = chat_policy(
                        partial_messages[:-1]
                        + [
                            dict(
                                role="assistant",
                                content=prefix,
                            )
                        ],
                        continue_final_message=True,
                        add_generation_prompt=False,
                        echo=True,
                        return_dict=True,
                        **continuation_kwargs,
                    )["choices"][0]
                    policy_message = policy_choice["message"]
                    response_text = policy_message.get("content") or prefix
                    generated_reasoning = policy_message.get(
                        "reasoning"
                    ) or policy_message.get("reasoning_content")
                    if generated_reasoning:
                        # Some servers parse generated text into the reasoning channel while the
                        # echoed prefix stays in content. Only close reasoning when the prefix
                        # itself still ends in that channel.
                        generated_content = response_text[len(prefix) :]
                        if generated_content:
                            prefix_message = adapter_policy.response_template.parse(
                                prefix,
                                messages=partial_messages[:-1],
                                tools=tools,
                            )
                            if (
                                "reasoning" in prefix_message
                                and "content" not in prefix_message
                                and "tool_calls" not in prefix_message
                                and prefix_message.get("finish_reason")
                                != "reasoning_end"
                            ):
                                generated_reasoning += (
                                    adapter_policy.response_template.reasoning_end_marker
                                )
                        response_text = prefix + generated_reasoning + generated_content
                    corrected_message = adapter_policy.response_template.parse(
                        response_text,
                        messages=partial_messages[:-1],
                        tools=tools,
                        finish_reason=policy_choice.get("finish_reason"),
                    )
                    policy_tool_calls = policy_message.get("tool_calls")
                    if policy_tool_calls and policy_tool_calls != [{}]:
                        # vLLM parses generated text before echoing the prefix, so these calls
                        # follow any calls already parsed from response_text.
                        corrected_tool_calls = corrected_message.get("tool_calls", [])
                        if corrected_tool_calls == [{}]:
                            corrected_tool_calls = []
                        corrected_message["tool_calls"] = (
                            corrected_tool_calls + policy_tool_calls
                        )
                        corrected_message = adapter_policy.response_template.parse(
                            adapter_policy.response_template.apply(corrected_message)[
                                "templated_prompt"
                            ],
                            messages=partial_messages[:-1],
                            tools=tools,
                            finish_reason=policy_choice.get("finish_reason"),
                        )
                    corrected_result["policy_model_name"] = chat_policy.model
                    corrected_messages = messages[:corrected_message_index] + [
                        corrected_message
                    ]
        else:  # make new message
            # Corrected result without correcting_model_name key means new message.
            corrected_messages = messages + [
                take_policy_message(
                    chat_policy(messages, return_dict=True, **policy_kwargs)
                )
            ]
            corrected_result["policy_model_name"] = chat_policy.model
            corrected_result["generate_new"] = True

        corrected_result["corrected_messages"] = corrected_messages[:]
        corrected_result["tools"] = tools
        # g()
        return corrected_result

    def iterative_correction_till_good(
        self,
        messages,
        chat_policy,
        max_rollouts=5,
        iid_sampling=False,
        adapter_policy=None,
        tools=None,
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
                    tools=tools,
                    applied_corrections=applied_corrections,
                )
            )

        while len(correction_steps) < max_rollouts:
            if not correction_steps:
                first_result = self.correct_and_rollout(
                    messages,
                    chat_policy,
                    iid_sampling=iid_sampling,
                    adapter_policy=adapter_policy,
                    tools=tools,
                )
                first_result["applied_corrections"] = applied_corrections
                correction_steps.append(first_result.copy())
                continue

            current_messages = correction_steps[-1]["corrected_messages"]
            corrected_result = self.correct_and_rollout(
                current_messages,
                chat_policy,
                iid_sampling=iid_sampling,
                adapter_policy=adapter_policy,
                tools=tools,
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
                elif correction["messages_location"].get("path_keys"):
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
        adapter_policy=None,
        tools=None,
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
                adapter_policy=adapter_policy,
                tools=tools,
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

    def test(
        self,
        chat_policy=None,
        adapter_policy=None,
        error_types=ERROR_TYPES,
    ):
        """Run model-backed smoke cases and return their raw results for inspection."""
        from boxx import mapmt

        if chat_policy is None:
            defaults = build_correcting_model_with_policy()
            chat_policy = defaults["chat_policy"]
            if adapter_policy is None:
                adapter_policy = defaults["adapter_policy"]

        correcteds = {}
        print("test error_types:", error_types)
        trajectories = get_test_trajectories()

        def f(error_type):
            trajectory = trajectories["error_type:" + error_type]
            msgs = deepcopy(trajectory["messages"])
            tools = deepcopy(trajectory.get("tools"))

            correcteds[error_type] = self.iterative_correction(
                msgs,
                chat_policy,
                rollout_num=3,
                mode="best_of_n",
                adapter_policy=adapter_policy,
                tools=tools,
                # iid_sampling=True,
            )

        mapmt(f, error_types, pool=len(error_types))
        return {"error_type:" + k: correcteds[k] for k in error_types}


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
            system_prompt_language=os.environ.get("FAR_SYSTEM_PROMPT_LANGUAGE"),
        )
    return CorrectingModel(
        chat_correcting,
        adapter,
    )


def build_correcting_model_with_policy(reasoning=True):
    import onpanda

    new_kwargs = dict(max_tokens=1536, temperature=0.8)
    if reasoning:
        correcting_model = build_reasoning_correcting_model()
        new_kwargs["model"] = os.environ.get(
            "POLICY_API_MODEL", "Qwen/Qwen3.5-35B-A3B"
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
    if os.environ.get("POLICY_API_BASE_URL"):
        chat_policy = mxlm.ChatAPI(
            os.environ.get("POLICY_API_BASE_URL"),
            os.environ.get("POLICY_API_KEY"),
            **chat_policy.default_kwargs,
        )
    # The policy renders its own response template, which is usually named after the policy, and
    # its own tokenizer decides how far a correction may reach into the continuation prefix.
    adapter_policy = onpanda.FindAndReplaceCorrectionAdapter(
        tokenizer=os.environ.get("POLICY_TOKENIZER_NAME_OR_PATH"),
        response_template=dict(
            name_or_path=os.environ.get(
                "POLICY_RESPONSE_TEMPLATE_NAME_OR_PATH", chat_policy.model
            )
        ),
    )
    return dict(
        correcting_model=correcting_model,
        chat_policy=chat_policy,
        adapter_policy=adapter_policy,
    )


if __name__ == "__main__":
    from boxx import *
    from onpanda.test_utils import (
        get_test_rejected_msgs1,
        get_test_trajectories,
    )

    _d = build_correcting_model_with_policy()
    correcting_model, chat_policy, adapter_policy = (
        _d["correcting_model"],
        _d["chat_policy"],
        _d["adapter_policy"],
    )

    if 10:
        correcteds = correcting_model.test(
            chat_policy=chat_policy,
            adapter_policy=adapter_policy,
        )
        tree(correcteds)
        print(savejson(correcteds, f"/tmp/{localTimeStr()}-correcteds.json"))

    msgs = [
        {"role": "user", "content": "5+7="},
        {"role": "assistant", "content": "32"},
        # {"role": "assistant", "content": "12"},
    ]
    msgs = get_test_rejected_msgs1()[0]
    tools = None
    trajectory = get_test_trajectories("bad_argument_arg2")
    msgs = trajectory["messages"]
    tools = trajectory.get("tools")

    # msgs = [{"role": "user", "content": "How many `1` in result of 652*8596"},]

    corrected = correcting_model.iterative_correction(
        msgs,
        chat_policy,
        rollout_num=3,
        mode="best_of_n",
        adapter_policy=_d["adapter_policy"],
        tools=tools,
        # iid_sampling=True,
    )
    tree(corrected)
