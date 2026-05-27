#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 16:56:53 2025

@author: DIYer22
"""
import copy
import json


BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL_NAME = "set_best_of_n_score"

BEST_OF_N_for_REASONING_CORRECTING_MODEL_SYSTEM_PROMPT = """
You are a strict best-of-n judge for reasoning model outputs.

You will receive a JSON array of candidate rollout records. Each candidate has:
- idx: a 1-based candidate id
- corrected_messages: the full conversation messages after that rollout/correction step
- assistant_content: the final assistant answer for that candidate
- correction_step: the full raw correction step metadata

Your task is to evaluate every candidate independently and assign a quality score.

Scoring rules:
- Give every candidate an integer score from 0 to 10, where 10 is best.
- Prefer answers that correctly solve the original user request.
- Prefer reliable reasoning, internal consistency, completeness, and a clear correct final answer.
- Penalize incorrect reasoning, arithmetic or factual errors, unsupported claims, incomplete answers, formatting failures, and answers that ignore the user request.
- Every candidate idx must appear exactly once in the result.
- comment is required for every candidate and should be one short sentence explaining the score.
- Do not output is_best. The caller will choose the best candidate by score.

You must call the tool named set_best_of_n_score.
Do not answer in plain text.
Do not omit any candidate.
""".strip()

BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL = {
    "type": "function",
    "function": {
        "name": BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL_NAME,
        "description": "Set best-of-n scores for all candidate rollout answers.",
        "parameters": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "description": "Mapping from 1-based candidate idx to score info.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "comment": {"type": "string"},
                        },
                        "required": ["score", "comment"],
                    },
                }
            },
            "required": ["scores"],
        },
    },
}


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
            prefill_logprobs = self.chat_correcting.prefill_logprobs(is_good_msgs)[-1][
                "prefill_logprobs"
            ]
            tree - prefill_logprobs
            print(is_good_prob)
            is_good_prob = e ** sum([d["logprob"] for d in prefill_logprobs])
            print(is_good_prob)
            import boxx.g
        return is_good_score

    def choose_best_of_n(self, correction_till_goods):
        best_of_n_score = {}
        for till_goods_idx, correction_till_good in enumerate(correction_till_goods):
            for step_idx, correction_step in enumerate(
                correction_till_good["correction_steps"]
            ):
                score = self.compute_is_good_score(
                    correction_step["corrected_messages"]
                )
                best_of_n_score[f"{till_goods_idx}/correction_steps/{step_idx}"] = score

        best_step_key = max(
            best_of_n_score, key=lambda key: best_of_n_score[key]["is_good_prob"]
        )
        till_goods_idx, step_idx = [
            int(i) for i in best_step_key.split("/correction_steps/")
        ]
        chosen_correction_step = correction_till_goods[till_goods_idx][
            "correction_steps"
        ][step_idx]
        delta_result = {**chosen_correction_step, "best_of_n_score": best_of_n_score}
        return delta_result

    def validate_best_of_n_for_reasoning_score(self, judge_response, candidate_num):
        try:
            message = judge_response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"judge_response missing choices[0].message: {repr(e)}"
            )

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            raise ValueError("judge_response has no tool_calls")

        last_error = None
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            if tool_name != BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL_NAME:
                continue
            try:
                arguments = function.get("arguments", "")
                if isinstance(arguments, str):
                    parsed_arguments = json.loads(arguments)
                elif isinstance(arguments, dict):
                    parsed_arguments = arguments
                else:
                    raise ValueError(
                        f"tool arguments must be str or dict, got {type(arguments)}"
                    )

                if not isinstance(parsed_arguments, dict):
                    raise ValueError("tool arguments JSON must be an object")

                scores_obj = parsed_arguments.get("scores", parsed_arguments)
                if not isinstance(scores_obj, dict):
                    raise ValueError("scores must be an object")

                scores_by_key = {str(k): v for k, v in scores_obj.items()}
                expected_keys = {str(i) for i in range(1, candidate_num + 1)}
                actual_keys = set(scores_by_key)
                if actual_keys != expected_keys:
                    raise ValueError(
                        "scores idx keys mismatch: "
                        f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
                    )

                normalized_scores = {}
                for idx in range(1, candidate_num + 1):
                    key = str(idx)
                    score_info = scores_by_key[key]
                    if not isinstance(score_info, dict):
                        raise ValueError(f"scores[{key}] must be an object")

                    score = score_info.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)):
                        raise ValueError(f"scores[{key}].score must be number")
                    if not 0 <= score <= 10:
                        raise ValueError(f"scores[{key}].score must be in [0, 10]")

                    if "comment" not in score_info:
                        raise ValueError(f"scores[{key}].comment is required")
                    comment = score_info["comment"]
                    if not isinstance(comment, str):
                        raise ValueError(f"scores[{key}].comment must be string")

                    normalized_score = dict(
                        is_best=False,
                        score=score,
                        comment=comment,
                    )
                    normalized_scores[key] = normalized_score

                best_idx = max(
                    range(1, candidate_num + 1),
                    key=lambda idx: (
                        normalized_scores[str(idx)]["score"],
                        -idx,
                    ),
                )
                normalized_scores[str(best_idx)]["is_best"] = True

                return dict(
                    scores=normalized_scores,
                    best_idx=best_idx,
                    tool_call=tool_call,
                )
            except Exception as e:
                last_error = e

        if last_error is not None:
            raise ValueError(
                "invalid set_best_of_n_score tool call: " + str(last_error)
            )
        raise ValueError(
            "judge_response has no tool_call named "
            f"{BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL_NAME}"
        )

    def compute_is_good_score_for_reasoning_correcting_model(self, messages):
        candidates = []
        for candidate in messages:
            candidates.append(
                dict(
                    idx=candidate["idx"],
                    step_key=candidate.get("step_key", ""),
                    corrected_messages=candidate.get("corrected_messages", []),
                    assistant_message=candidate.get("assistant_message", {}),
                    assistant_content=candidate.get("assistant_content", ""),
                    correction_step=candidate.get("correction_step", {}),
                )
            )

        judge_messages = [
            {
                "role": "system",
                "content": BEST_OF_N_for_REASONING_CORRECTING_MODEL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"candidates": candidates},
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            },
        ]
        old_parser = getattr(self.chat_correcting, "parser", None)
        has_parser = hasattr(self.chat_correcting, "parser")
        if has_parser:
            self.chat_correcting.parser = None
        try:
            judge_response = self.chat_correcting(
                judge_messages,
                return_dict=True,
                tools=[copy.deepcopy(BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL)],
                tool_choice={
                    "type": "function",
                    "function": {
                        "name": BEST_OF_N_for_REASONING_CORRECTING_MODEL_TOOL_NAME
                    },
                },
            )
        finally:
            if has_parser:
                self.chat_correcting.parser = old_parser
        validated = self.validate_best_of_n_for_reasoning_score(
            judge_response, len(candidates)
        )
        return dict(
            scores=validated["scores"],
            best_idx=validated["best_idx"],
            tool_call=validated["tool_call"],
            judge_response=judge_response,
        )

    def choose_best_of_n_for_reasoning_correcting_model(self, correction_till_goods):
        candidates = []
        for till_goods_idx, correction_till_good in enumerate(correction_till_goods):
            for step_idx, correction_step in enumerate(
                correction_till_good["correction_steps"]
            ):
                corrected_messages = copy.deepcopy(
                    correction_step.get("corrected_messages", [])
                )
                assistant_message = (
                    copy.deepcopy(corrected_messages[-1])
                    if corrected_messages
                    else {}
                )
                assistant_content = assistant_message.get("content", "")
                if not isinstance(assistant_content, str):
                    assistant_content = json.dumps(
                        assistant_content, ensure_ascii=False, default=str
                    )
                candidates.append(
                    dict(
                        idx=len(candidates) + 1,
                        step_key=f"{till_goods_idx}/correction_steps/{step_idx}",
                        corrected_messages=corrected_messages,
                        assistant_message=assistant_message,
                        assistant_content=assistant_content,
                        correction_step=copy.deepcopy(correction_step),
                    )
                )

        if not candidates:
            raise ValueError("No correction steps found for best_of_n candidates.")

        attempts = []
        judge_result = None
        last_error = None
        max_attempts = 5

        for attempt_idx in range(1, max_attempts + 1):
            try:
                judge_result = self.compute_is_good_score_for_reasoning_correcting_model(
                    candidates
                )
                attempts.append(dict(attempt=attempt_idx, ok=True))
                break
            except Exception as e:
                last_error = e
                attempts.append(
                    dict(attempt=attempt_idx, ok=False, error=str(e))
                )

        if judge_result is None:
            raise ValueError(
                "best_of_n_for_reasoning_correcting_model failed after "
                f"{max_attempts} attempts: {last_error}"
            )

        best_idx = judge_result["best_idx"]
        chosen_candidate = candidates[best_idx - 1]
        chosen_correction_step = chosen_candidate["correction_step"]
        delta_result = {
            **chosen_correction_step,
            "best_of_n_for_reasoning_correcting_model_score": judge_result["scores"],
            "best_of_n_for_reasoning_correcting_model_best_idx": best_idx,
            "best_of_n_for_reasoning_correcting_model_best_key": chosen_candidate[
                "step_key"
            ],
            "best_of_n_for_reasoning_correcting_model_candidates": candidates,
            "best_of_n_for_reasoning_correcting_model_judge_response": judge_result[
                "judge_response"
            ],
            "best_of_n_for_reasoning_correcting_model_attempts": attempts,
        }
        return delta_result


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
