#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze panda score json with unified summary and full-score_results drilldown.
"""

import argparse
import difflib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from onpanda.correcting_model.panda_score_mixin import DEFAULT_TMP_PANDA_SCORE_PATH

DEFAULT_GENERATE_MODULE = "onpanda.correcting_model.panda_score_mixin"
DEFAULT_JSON_PATH = DEFAULT_TMP_PANDA_SCORE_PATH


def _prepare_default_json_path() -> Path:
    cmd = [
        sys.executable,
        "-W",
        "ignore::RuntimeWarning:runpy",
        "-m",
        DEFAULT_GENERATE_MODULE,
    ]
    print(f"[INFO] --json-path not set. Run: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print("[INFO] panda_score generation finished\n" + "-" * 72, flush=True)
    return DEFAULT_JSON_PATH.resolve()


def _parse_feedback_lines(feedback: str) -> Dict[str, str]:
    parsed = {
        "format_feedback": "",
        "location_feedback": "",
        "replacement_feedback": "",
    }
    for line in str(feedback).splitlines():
        if line.startswith("format_feedback: "):
            parsed["format_feedback"] = line.split(": ", 1)[1]
        elif line.startswith("location_feedback: "):
            parsed["location_feedback"] = line.split(": ", 1)[1]
        elif line.startswith("replacement_feedback: "):
            parsed["replacement_feedback"] = line.split(": ", 1)[1]
    return parsed


def _short_text(text: Any, limit: int = 120) -> str:
    s = str(text).replace("\n", "↳")
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _fmt_reward_num(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _summary_by_correcting_model(panda_score: Dict[str, Any]) -> str:
    # Call CorrectingModel.summary_panda_score directly, without initializing
    # model runtime dependencies.
    from onpanda.correcting_model.correcting_model import CorrectingModel

    correcting_model = object.__new__(CorrectingModel)
    return correcting_model.summary_panda_score(panda_score)


def _parse_summary_text(summary_text: str) -> Dict[str, str]:
    summary = {}
    for line in summary_text.splitlines():
        if ": " not in line:
            continue
        k, v = line.split(": ", 1)
        summary[k] = v
    return summary


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_one(value: Any, tol: float = 1e-12) -> bool:
    return abs(_to_float(value, 0.0) - 1.0) <= tol


def _joint_key(score_result: Dict[str, Any]) -> Tuple[float, float, float]:
    panda_score = score_result.get("panda_score", {})
    return (
        _to_float(panda_score.get("format_score", 0.0)),
        _to_float(panda_score.get("location_score", 0.0)),
        _to_float(panda_score.get("replacement_score", 0.0)),
    )


def _is_full_panda_match(score_result: Dict[str, Any]) -> bool:
    panda_score = score_result.get("panda_score", {})
    return (
        _is_one(panda_score.get("format_score"))
        and _is_one(panda_score.get("location_score"))
        and _is_one(panda_score.get("replacement_score"))
    )


def _reward_triplet_key(score_result: Dict[str, Any]) -> Tuple[float, float, float]:
    reward = score_result.get("reward_with_feedback", {})
    return (
        _to_float(reward.get("format_reward", 0.0)),
        _to_float(reward.get("location_reward", 0.0)),
        _to_float(reward.get("replacement_reward", 0.0)),
    )


def _build_sample(score_result: Dict[str, Any]) -> Dict[str, Any]:
    reward_with_feedback = score_result.get("reward_with_feedback", {})
    parsed_feedback = _parse_feedback_lines(reward_with_feedback.get("feedback", ""))
    gt = score_result.get("gt_find_and_replace", {})
    pred = score_result.get("find_and_replace", {})
    info = score_result.get("info", {})
    correction_response = score_result.get("correction_result", {}).get(
        "correction_response", {}
    )
    choices = correction_response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")

    return {
        "panda_path": info.get("panda_path", ""),
        "far_correction_data_idx": info.get("far_correction_data_idx"),
        "onpanda_uuid": info.get("onpanda", {}).get("uuid"),
        "gt_is_good": bool(gt.get("is_good")),
        "pred_is_good": bool(pred.get("is_good")),
        "format_feedback": parsed_feedback["format_feedback"],
        "location_feedback": parsed_feedback["location_feedback"],
        "replacement_feedback": parsed_feedback["replacement_feedback"],
        "pred_location_text": _short_text(pred.get("location_text", "")),
        "gt_location_text": _short_text(gt.get("location_text", "")),
        "pred_replacement_token": _short_text(pred.get("replacement_token", "")),
        "gt_replacement_token": _short_text(gt.get("replacement_token", "")),
        "pred_far_text": _short_text(pred.get("far_text", ""), limit=260),
        "gt_far_text": _short_text(gt.get("far_text", ""), limit=260),
        "context": _build_location_context(score_result),
        "reward_with_feedback": reward_with_feedback,
        "panda_score": score_result.get("panda_score", {}),
        "reasoning": reasoning,
    }


def _infer_verifier_special_tokens(find_and_replace: Dict[str, Any]) -> Dict[str, str]:
    special_tokens: Dict[str, str] = {}
    far_text = str(find_and_replace.get("far_text", ""))
    if far_text.startswith("<|") and "|>" in far_text:
        split_token = far_text[: far_text.find("|>") + 2]
        if split_token and far_text.endswith(split_token):
            special_tokens["split"] = split_token

    location_text = str(find_and_replace.get("location_text", ""))
    replacement_token = str(find_and_replace.get("replacement_token", ""))
    if "<|fim_suffix|>" in location_text or "<|fim_suffix|>" in replacement_token:
        special_tokens["stop"] = "<|fim_suffix|>"

    if bool(find_and_replace.get("is_good")) and "split" in special_tokens:
        split_token = special_tokens["split"]
        if far_text.startswith(split_token) and far_text.endswith(split_token):
            is_good_token = far_text[len(split_token) : -len(split_token)]
            if is_good_token:
                special_tokens["is_good"] = is_good_token
    return special_tokens


def _build_context_text(
    text: str, char_index: int, patch_length: int, context_window: int = 24
) -> str:
    if not text:
        return ""
    char_index = min(max(char_index, 0), len(text))
    patch_length = max(patch_length, 1)
    patch_end = min(char_index + patch_length, len(text))
    if patch_end <= char_index:
        patch_end = min(char_index + 1, len(text))
    start = max(0, char_index - context_window)
    end = min(len(text), patch_end + context_window)
    context = (
        f"{text[start:char_index]}【{text[char_index:patch_end]}】{text[patch_end:end]}"
    )
    return _short_text(context, limit=160)


def _build_fuzzy_context(
    verifier: Any, messages: List[Dict[str, Any]], location_text: str
):
    if not location_text:
        return ""

    target_len = len(location_text)
    best_ratio = -1.0
    best_text = ""
    best_char_index = 0
    best_patch_length = 1

    for templated_location in verifier._iter_assistant_templated_prompts(messages):
        text = templated_location["templated_prompt"]
        if not text:
            continue
        if len(text) <= target_len:
            ratio = difflib.SequenceMatcher(None, location_text, text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = text
                best_char_index = 0
                best_patch_length = max(len(text), 1)
            continue

        step = max(target_len // 4, 1)
        last_start = len(text) - target_len
        starts = list(range(0, last_start + 1, step))
        if starts[-1] != last_start:
            starts.append(last_start)
        for start in starts:
            candidate = text[start : start + target_len]
            ratio = difflib.SequenceMatcher(None, location_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = text
                best_char_index = start
                best_patch_length = target_len

    if best_ratio < 0.0:
        return ""
    context = _build_context_text(
        best_text,
        char_index=best_char_index,
        patch_length=best_patch_length,
    )
    return f"{context} (fuzzy={best_ratio:.3f})"


def _build_location_context(score_result: Dict[str, Any]) -> str:
    from onpanda.correcting_model.far_text_parse import FindAndReplaceCodecMixin

    trimmed_messages = score_result.get("correction_result", {}).get(
        "trimmed_messages", []
    )
    if not trimmed_messages:
        return ""

    find_and_replace = score_result.get("find_and_replace", {})
    codec = FindAndReplaceCodecMixin(
        special_tokens=_infer_verifier_special_tokens(find_and_replace)
    )
    correction = codec.parse_and_locate(
        trimmed_messages, str(find_and_replace.get("far_text", ""))
    )
    parsed_find_and_replace = correction.get("find_and_replace", {})
    messages_location = correction.get("messages_location", {})
    locate_reward = correction.get("reward_with_feedback", {})
    format_reward = score_result.get("reward_with_feedback", {}).get("format_reward")

    if (
        messages_location.get("not_found")
        and locate_reward.get("find_feedback") == "location_text not found"
    ):
        return _build_fuzzy_context(
            codec,
            trimmed_messages,
            location_text=str(parsed_find_and_replace.get("location_text", "")),
        )

    if _is_one(format_reward) and not messages_location.get("not_found"):
        path_keys = messages_location.get("path_keys")
        char_index = messages_location.get("char_index")
        if not isinstance(path_keys, list) or not isinstance(char_index, int):
            return ""

        text = trimmed_messages
        for key in path_keys:
            text = text[key]
        text = codec._content_to_text(text)
        return _build_context_text(
            text,
            char_index=char_index,
            patch_length=len(str(parsed_find_and_replace.get("location_text", ""))),
        )
    return ""


def _score_result_sort_key(score_result: Dict[str, Any]) -> Tuple[str, float]:
    info = score_result.get("info", {})
    panda_path = str(info.get("panda_path", ""))
    idx = info.get("far_correction_data_idx")
    if idx is None:
        idx_num = float("inf")
    else:
        idx_num = _to_float(idx, float("inf"))
    return panda_path, idx_num


def _render_context_for_cli(context: str) -> str:
    left = context.find("【")
    if left < 0:
        return context
    right = context.find("】", left + 1)
    if right < 0:
        return context
    return (
        f"{context[:left]}"
        f"\033[31m{context[left + 1:right]}\033[0m"
        f"{context[right + 1:]}"
    )


def _is_both_is_good(score_result: Dict[str, Any]) -> bool:
    gt_is_good = bool(score_result.get("gt_find_and_replace", {}).get("is_good"))
    pred_is_good = bool(score_result.get("find_and_replace", {}).get("is_good"))
    return gt_is_good and pred_is_good


def analyze(
    panda_score: Dict[str, Any],
    sample_limit: int = 10,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    score_results = panda_score.get("score_results", [])
    if not score_results:
        return {"error": "No score_results found in panda_score json."}

    rng = random.Random(random_seed)

    summary_text = _summary_by_correcting_model(panda_score)
    score_result_num = len(score_results)

    joint_counter = Counter(_joint_key(x) for x in score_results)
    joint_gt_good_counter = Counter(
        _joint_key(x)
        for x in score_results
        if bool(x.get("gt_find_and_replace", {}).get("is_good"))
    )

    reward_triplet_counter = Counter(_reward_triplet_key(x) for x in score_results)
    reward_triplet_gt_good_counter = Counter(
        _reward_triplet_key(x)
        for x in score_results
        if bool(x.get("gt_find_and_replace", {}).get("is_good"))
    )
    reward_triplet_gt_good_wrong_counter = Counter(
        _reward_triplet_key(x)
        for x in score_results
        if bool(x.get("gt_find_and_replace", {}).get("is_good"))
        and not _is_full_panda_match(x)
    )
    reward_triplet_samples: Dict[Tuple[float, float, float], List[Dict[str, Any]]] = {}

    reward_triplet_items: Dict[Tuple[float, float, float], List[Dict[str, Any]]] = {}
    for item in score_results:
        triplet_key = _reward_triplet_key(item)
        reward_triplet_items.setdefault(triplet_key, []).append(item)

    # Sample randomly within each triplet group to avoid biased display order.
    # Prefer informative cases by excluding "both is_good" from displayed samples
    # when the group still has other candidates.
    for triplet_key, items in reward_triplet_items.items():
        informative_items = [item for item in items if not _is_both_is_good(item)]
        selected = informative_items if informative_items else list(items)
        rng.shuffle(selected)
        selected = sorted(selected[:sample_limit], key=_score_result_sort_key)
        reward_triplet_samples[triplet_key] = [_build_sample(item) for item in selected]

    top_n = min(10, len(joint_counter))
    top_joint = [
        {
            "format_score": k[0],
            "location_score": k[1],
            "replacement_score": k[2],
            "count": v,
            "rate_on_all": v / score_result_num,
            "gt_good_count": joint_gt_good_counter.get(k, 0),
            "gt_not_good_count": v - joint_gt_good_counter.get(k, 0),
        }
        for k, v in joint_counter.most_common(top_n)
    ]
    reward_triplet_groups = [
        {
            "format_reward": k[0],
            "location_reward": k[1],
            "replacement_reward": k[2],
            "total_reward": sum(k),
            "count": v,
            "rate_on_all": v / score_result_num,
            "gt_good_count": reward_triplet_gt_good_counter.get(k, 0),
            "gt_not_good_count": v - reward_triplet_gt_good_counter.get(k, 0),
            "gt_good_wrong_count": reward_triplet_gt_good_wrong_counter.get(k, 0),
            "samples": reward_triplet_samples.get(k, []),
        }
        for k, v in sorted(
            reward_triplet_counter.items(),
            key=lambda kv: (sum(kv[0]), kv[0]),
        )
    ]

    gt_good_wrong = [
        x
        for x in score_results
        if bool(x.get("gt_find_and_replace", {}).get("is_good"))
        and not _is_full_panda_match(x)
    ]
    gt_good_wrong_random = list(gt_good_wrong)
    rng.shuffle(gt_good_wrong_random)
    gt_good_wrong_random = sorted(
        gt_good_wrong_random[:sample_limit], key=_score_result_sort_key
    )
    gt_good_wrong_samples = [_build_sample(item) for item in gt_good_wrong_random]

    return {
        "summary_text": summary_text,
        "summary": _parse_summary_text(summary_text),
        "score_result_num": score_result_num,
        "gt_good_wrong_num": len(gt_good_wrong),
        "top_joint_distribution": top_joint,
        "reward_triplet_groups": reward_triplet_groups,
        "gt_good_wrong_samples": gt_good_wrong_samples,
        "sample_limit_per_triplet": sample_limit,
        "random_seed": random_seed,
    }


def _print_sample_block(sample: Dict[str, Any], idx: int) -> None:
    print(
        f"\n[{idx}] panda_path={sample['panda_path']} idx={sample['far_correction_data_idx']}"
    )
    print(
        f"    gt_is_good={sample['gt_is_good']} pred_is_good={sample['pred_is_good']}"
    )
    print(
        f"    feedback: fmt={sample['format_feedback']} | "
        f"loc={sample['location_feedback']} | rep={sample['replacement_feedback']}"
    )
    if sample.get("context"):
        print(f"     ctx {_render_context_for_cli(str(sample['context']))}")
    print(f"    pred {sample['pred_far_text']}")
    print(f"      gt {sample['gt_far_text']}")
    reward = sample.get("reward_with_feedback", {})
    print(
        "    reward: "
        f"final={_fmt_reward_num(reward.get('final_reward'))} "
        f"format={_fmt_reward_num(reward.get('format_reward'))} "
        f"location={_fmt_reward_num(reward.get('location_reward'))} "
        f"replacement={_fmt_reward_num(reward.get('replacement_reward'))}"
    )
    if reward.get("format_feedback"):
        print(f"    reward_format_feedback: {reward['format_feedback']}")


def print_human_report(result: Dict[str, Any]) -> None:
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    print("=== Summary ===")
    print(result["summary_text"])

    print(
        "\n=== Top Joint Distribution on all score_results (format, location, replacement) ==="
    )
    for row in result["top_joint_distribution"]:
        print(
            f"({row['format_score']}, {row['location_score']}, {row['replacement_score']}) "
            f"count={row['count']} rate={row['rate_on_all']:.4f} "
            f"gt_good={row['gt_good_count']} gt_not_good={row['gt_not_good_count']}"
        )

    sample_limit = result.get("sample_limit_per_triplet", 10)
    print(
        "\n=== Reward Triplet Groups on all score_results "
        f"(sorted by total reward asc, max {sample_limit} samples each) ==="
    )
    for group in result["reward_triplet_groups"]:
        print(
            f"\n(format_reward={group['format_reward']}, "
            f"location_reward={group['location_reward']}, "
            f"replacement_reward={group['replacement_reward']}) "
            f"total={group['total_reward']:.3f} "
            f"count={group['count']} rate={group['rate_on_all']:.4f} "
            f"gt_good={group['gt_good_count']} "
            f"gt_not_good={group['gt_not_good_count']} "
            f"gt_good_wrong={group['gt_good_wrong_count']}"
        )
        for i, sample in enumerate(group["samples"], start=1):
            _print_sample_block(sample, i)

    print("\n=== GT is good but wrong (panda score triplet != 1.0,1.0,1.0) ===")
    print(
        f"count={result['gt_good_wrong_num']} "
        f"(showing up to {sample_limit} samples)"
    )
    for i, sample in enumerate(result["gt_good_wrong_samples"], start=1):
        _print_sample_block(sample, i)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Analyze panda_score json with full score_results grouping.")
    )
    parser.add_argument(
        "--json-path",
        default="",
        help=(
            "Path to *.panda_score.json. "
            "If empty, run `python -m onpanda.correcting_model.panda_score_mixin` "
            "and read the generated file in system temp dir."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="How many samples to print/save per reward triplet combination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sample display selection. Omit for random each run.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional output path to save full analysis result as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.json_path:
        json_path = Path(args.json_path).expanduser().resolve()
    else:
        json_path = _prepare_default_json_path()
    print(f"[INFO] analyze json: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        panda_score = json.load(f)

    result = analyze(panda_score, sample_limit=args.sample_limit, random_seed=args.seed)
    print_human_report(result)

    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nAnalysis JSON saved to: {save_path}")
