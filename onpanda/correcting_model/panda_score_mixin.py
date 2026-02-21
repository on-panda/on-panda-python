#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 04:52:00 2026

@author: DIYer22
"""
import os
import mximport

with mximport.inpkg():
    from ..utils import (
        remove_msgs_after_last_response_role,
        slim_json_strings,
    )


class PandaScoreMixin:
    @staticmethod
    def _mean(nums):
        return sum(nums) / len(nums)

    def eval_panda_score(self, panda_json_path):
        """
        Evaluate correcting quality on panda json file(s).

        Returned summary metrics:
        - format_score: mean format reward on all samples
        - location_score: mean location reward on all samples
        - replacement_score: mean replacement reward on all samples
        - is_good_cls_score: accuracy of predicting is_good on all samples
        - location_on_not_good_score: mean location reward on not_good samples
        - replacement_on_not_good_score: mean replacement reward on not_good samples
        - replacement_on_true_location_score: mean replacement reward on not_good
          samples where location is correct
        - not_good_num: number of not_good samples
        - true_location_num: number of not_good samples with correct location
        """
        from onpanda import PandaTree

        if isinstance(panda_json_path, (list, tuple)):
            panda_score_list = [self.eval_panda_score(path) for path in panda_json_path]
            return self.aggregation_panda_score(panda_score_list)

        format_scores = []
        location_scores = []
        replacement_scores = []
        is_good_cls_scores = []
        location_on_not_good_scores = []
        replacement_on_not_good_scores = []
        replacement_on_true_location_scores = []
        not_good_num = 0
        true_location_num = 0
        score_results = []

        panda_tree = PandaTree(panda_json_path, tokenizer=self.adapter.tokenizer)
        far_correction_datas = panda_tree.build_far_correction_data_v1(self.adapter)
        for far_correction_data_idx, far_correction_data in enumerate(
            far_correction_datas
        ):
            messages = remove_msgs_after_last_response_role(far_correction_data[:-2])
            gt_correction = far_correction_data[-1]["correcting"]

            correcting_result = self.correcting(messages)
            far_text = correcting_result["far_text"]
            reward_result = self.adapter.verifier.compute_reward(
                messages,
                far_text,
                gt_correction,
            )
            reward_with_feedback = reward_result["reward_with_feedback"]

            gt_is_good = bool(
                gt_correction["find_and_replace"].get("is_good")
                or gt_correction["messages_location"].get("is_good")
            )
            pred_is_good = bool(correcting_result.get("is_good"))
            panda_score = dict(
                format_score=reward_with_feedback["format_reward"],
                location_score=reward_with_feedback["location_reward"],
                replacement_score=reward_with_feedback["replacement_reward"],
                is_good_cls_score=1.0 if gt_is_good == pred_is_good else 0.0,
            )
            reward_result["gt_find_and_replace"] = gt_correction["find_and_replace"]
            reward_result["panda_score"] = panda_score
            reward_result["info"] = dict(
                panda_path=panda_json_path,
                far_correction_data_idx=far_correction_data_idx,
                onpanda=far_correction_data[0].get("onpanda"),
            )
            reward_result["correcting_result"] = {
                k: correcting_result[k] for k in ["corrector_response"]
            }
            reward_result["correcting_result"]["trimmed_messages"] = slim_json_strings(
                messages
            )
            score_results.append(reward_result)

            format_scores.append(panda_score["format_score"])
            location_scores.append(panda_score["location_score"])
            replacement_scores.append(panda_score["replacement_score"])
            is_good_cls_scores.append(panda_score["is_good_cls_score"])

            if not gt_is_good:
                not_good_num += 1
                location_on_not_good_scores.append(panda_score["location_score"])
                replacement_on_not_good_scores.append(panda_score["replacement_score"])
                if panda_score["location_score"] == 1.0:
                    true_location_num += 1
                    replacement_on_true_location_scores.append(
                        panda_score["replacement_score"]
                    )
        # import boxx.g
        panda_score = dict(
            format_score=self._mean(format_scores),
            location_score=self._mean(location_scores),
            replacement_score=self._mean(replacement_scores),
            is_good_cls_score=self._mean(is_good_cls_scores),
            location_on_not_good_score=(
                self._mean(location_on_not_good_scores) if not_good_num else 0.0
            ),
            replacement_on_not_good_score=(
                self._mean(replacement_on_not_good_scores) if not_good_num else 0.0
            ),
            replacement_on_true_location_score=(
                self._mean(replacement_on_true_location_scores)
                if true_location_num
                else 0.0
            ),
            not_good_num=not_good_num,
            true_location_num=true_location_num,
            score_result_num=len(score_results),
            panda_json_num=1,
            score_results=score_results,
        )
        return panda_score

    def aggregation_panda_score(self, panda_scores):
        score_results = []
        score_result_num = 0
        panda_json_num = 0
        not_good_num = 0
        true_location_num = 0

        format_score_sum = 0.0
        location_score_sum = 0.0
        replacement_score_sum = 0.0
        is_good_cls_score_sum = 0.0
        location_on_not_good_score_sum = 0.0
        replacement_on_not_good_score_sum = 0.0
        replacement_on_true_location_score_sum = 0.0

        for panda_score in panda_scores:
            score_result_num += panda_score["score_result_num"]
            panda_json_num += panda_score["panda_json_num"]
            not_good_num += panda_score["not_good_num"]
            true_location_num += panda_score["true_location_num"]

            format_score_sum += (
                panda_score["format_score"] * panda_score["score_result_num"]
            )
            location_score_sum += (
                panda_score["location_score"] * panda_score["score_result_num"]
            )
            replacement_score_sum += (
                panda_score["replacement_score"] * panda_score["score_result_num"]
            )
            is_good_cls_score_sum += (
                panda_score["is_good_cls_score"] * panda_score["score_result_num"]
            )
            location_on_not_good_score_sum += (
                panda_score["location_on_not_good_score"] * panda_score["not_good_num"]
            )
            replacement_on_not_good_score_sum += (
                panda_score["replacement_on_not_good_score"]
                * panda_score["not_good_num"]
            )
            replacement_on_true_location_score_sum += (
                panda_score["replacement_on_true_location_score"]
                * panda_score["true_location_num"]
            )
            score_results.extend(panda_score["score_results"])

        panda_score = dict(
            format_score=format_score_sum / score_result_num,
            location_score=location_score_sum / score_result_num,
            replacement_score=replacement_score_sum / score_result_num,
            is_good_cls_score=is_good_cls_score_sum / score_result_num,
            location_on_not_good_score=(
                location_on_not_good_score_sum / not_good_num if not_good_num else 0.0
            ),
            replacement_on_not_good_score=(
                replacement_on_not_good_score_sum / not_good_num
                if not_good_num
                else 0.0
            ),
            replacement_on_true_location_score=(
                replacement_on_true_location_score_sum / true_location_num
                if true_location_num
                else 0.0
            ),
            not_good_num=not_good_num,
            true_location_num=true_location_num,
            score_result_num=score_result_num,
            panda_json_num=panda_json_num,
            score_results=score_results,
        )
        return panda_score

    def summary_panda_score(self, panda_score):
        return "\n".join(
            [
                (
                    f"{k}: {(panda_score[k])*100:04.1f}%"
                    if isinstance(panda_score[k], float)
                    else f"{k}: {panda_score[k]}"
                )
                for k in (
                    "format_score",
                    "location_score",
                    "replacement_score",
                    "is_good_cls_score",
                    "location_on_not_good_score",
                    "replacement_on_not_good_score",
                    "replacement_on_true_location_score",
                    "not_good_num",
                    "true_location_num",
                    "score_result_num",
                    "panda_json_num",
                )
            ]
        )


if __name__ == "__main__":
    from boxx import tree
    from glob import glob

    with mximport.inpkg():
        from .correcting_model import build_test_correcting_model

    correcting_model = build_test_correcting_model(tokenizer=None)
    test_json_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "../../../on-panda-example-data/panda_json/2025-04-12_Chinese_acrostic_poem_藏头诗_tokenizer-step2.panda.json",
        )
    ]

    # correcting_model = build_test_correcting_model(tokenizer="/home/yl/audio/asset/tokenizer/step1f/")
    # test_json_paths = sorted(glob("/home/yl/audio/jili_tracelogs_bmk/step1f_tracelogs2_batch5_jili_bmk/*.panda.json"))[:13]
    panda_score = correcting_model.eval_panda_score(test_json_paths)
    score_results = panda_score["score_results"]
    tree(score_results, deep=3)
    print(correcting_model.summary_panda_score(panda_score))
