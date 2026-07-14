#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistics CLI for onPanda multimodal panda-json data."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as py_statistics
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mxlm
import onpanda


DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_TOKENIZER_FALLBACKS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"
HOUR_MS = 60 * 60 * 1000
DEFAULT_MAX_ANNOTATION_HOURS = 7.0
DEFAULT_PASTE_THRESHOLD = 0.8
COMPLETE_ROLLOUT_FINISH_REASONS = {"stop", "tool_calls"}
SUPPORTED_MULTIMODAL_TYPES = {
    "vlm": {"image", "video"},
    "audio": {"audio"},
    "agentic": {"image"},
}


def resolve_paths(sources: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for source in sources:
        expanded = os.path.expanduser(source)
        if glob.has_magic(expanded):
            matched = [Path(path) for path in glob.glob(expanded, recursive=True)]
        else:
            path = Path(expanded)
            if not path.exists():
                raise FileNotFoundError(f"Statistics source does not exist: {source}")
            matched = sorted(path.rglob("*.panda.json")) if path.is_dir() else [path]
        if not matched:
            raise FileNotFoundError(f"Statistics source matched no files: {source}")
        paths.extend(matched)

    return sorted({Path(os.path.abspath(path)) for path in paths})


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    candidates = [name_or_path]
    if name_or_path in {"qwen2.5", "qwen25", DEFAULT_TOKENIZER}:
        candidates = list(QWEN25_TOKENIZER_FALLBACKS)

    errors = []
    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        except (OSError, ValueError) as exc:
            errors.append(f"{candidate} local: {type(exc).__name__}: {exc}")
    raise RuntimeError("Failed to load tokenizer:\n" + "\n".join(errors))


class TokenCounter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache: Dict[str, int] = {}

    def count(self, text: Any) -> int:
        if text is None:
            return 0
        text = str(text)
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        token_num = len(self.tokenizer.encode(text, add_special_tokens=False))
        if len(self._cache) < 200000:
            self._cache[text] = token_num
        return token_num

    def fork_index(self, text: str, char_index: int) -> int:
        """Return the token index containing the first differing character."""
        text = str(text)
        if char_index <= 0:
            return 0
        if char_index >= len(text):
            return self.count(text)

        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        prefix_tokens = self.tokenizer.encode(
            text[:char_index], add_special_tokens=False
        )
        suffix_tokens = self.tokenizer.encode(
            text[char_index:], add_special_tokens=False
        )
        if prefix_tokens and prefix_tokens + suffix_tokens != full_tokens:
            return len(prefix_tokens) - 1
        return len(prefix_tokens)

    def count_sources(self, text: str, sources: Sequence[str]) -> Tuple[Counter, int]:
        assert len(text) == len(sources)
        offsets = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )["offset_mapping"]
        source_counts: Counter = Counter()
        mixed_source_tokens = 0
        priority = {"model": 0, "candidate": 1, "manual": 2}
        for start, end in offsets:
            overlaps = Counter(sources[start:end])
            assert overlaps
            if len(overlaps) > 1:
                mixed_source_tokens += 1
            source = max(overlaps, key=lambda key: (overlaps[key], priority[key]))
            source_counts[source] += 1
        return source_counts, mixed_source_tokens


def _last_response_message(
    messages: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message
    return None


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)
    return str(content)


def _recover_response_hashes(data: Dict[str, Any]) -> None:
    """Resolve response hashes without materializing large prompt assets."""
    hash_map = data.get("hash_map") or {}
    for collection in ("dialogs", "deleted_dialogs"):
        for dialog in (data.get(collection) or {}).values():
            for message in dialog.get("messages") or []:
                if message.get("role") != "assistant":
                    continue
                for field in ("reasoning", "content", "tool_calls"):
                    value = message.get(field)
                    if not isinstance(value, str):
                        continue
                    match = re.fullmatch(onpanda.HASH_TEMPLATE_REGEX, value)
                    if match:
                        message[field] = hash_map[match.group(1)]


def _restore_agentic_message_roles(data: Dict[str, Any]) -> Counter:
    """Restore roles omitted from agentic snapshots using recorded evidence."""
    messages = [
        message
        for collection in ("dialogs", "deleted_dialogs")
        for dialog in (data.get(collection) or {}).values()
        for message in dialog.get("messages") or []
    ]
    missing_role_messages = [
        message for message in messages if message.get("role") is None
    ]
    if not missing_role_messages:
        return Counter()

    restored: Counter = Counter()
    unresolved_messages = []
    for message in missing_role_messages:
        if message.get("reasoning") is not None or message.get("tool_calls"):
            message["role"] = "assistant"
            restored["assistant"] += 1
        else:
            unresolved_messages.append(message)
    if not unresolved_messages:
        return restored

    def message_signature(message: Dict[str, Any]) -> str:
        return json.dumps(
            {key: value for key, value in message.items() if key != "role"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    recorded_roles: Dict[str, set] = defaultdict(set)
    for message in messages:
        if message.get("role") is not None:
            recorded_roles[message_signature(message)].add(message["role"])

    for message in unresolved_messages:
        roles = recorded_roles[message_signature(message)]
        if len(roles) != 1:
            raise ValueError(
                "Cannot recover an agentic message role from recorded snapshots: "
                f"{message}"
            )
        role = next(iter(roles))
        message["role"] = role
        restored[role] += 1
    return restored


def _serialize_response(
    message: Dict[str, Any],
) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    reasoning_present = message.get("reasoning") is not None
    reasoning = content_to_text(message.get("reasoning"))
    content = content_to_text(message.get("content"))
    tool_calls = message.get("tool_calls") or []

    if not reasoning_present and not tool_calls:
        tool_start = content.find("<|tool_calls_section_begin|>")
        if tool_start < 0:
            tool_start = len(content)
        reasoning_end = content.find("</think>")
        if reasoning_end < 0 or reasoning_end >= tool_start:
            reasoning_end = 0
        else:
            reasoning_end += len("</think>")
        return content, {
            "reasoning": (0, reasoning_end),
            "content": (reasoning_end, tool_start),
            "tool_call": (tool_start, len(content)),
        }

    text = reasoning
    if reasoning_present and (
        content or tool_calls or message.get("finish_reason") == "reasoning_end"
    ):
        text += "</think>"
    ranges = {"reasoning": (0, len(text))}

    content_start = len(text)
    text += content
    ranges["content"] = (content_start, len(text))

    tool_call_start = len(text)
    if tool_calls:
        assert isinstance(tool_calls, list)
        text += "<|tool_calls_section_begin|>"
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if (
                not tool_call.get("id")
                or not function.get("name")
                or "arguments" not in function
            ):
                text += json.dumps(
                    tool_call, ensure_ascii=False, separators=(",", ":")
                )
                continue
            arguments = function["arguments"]
            if not isinstance(arguments, str):
                arguments = json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                )
            text += (
                "<|tool_call_begin|>"
                + tool_call["id"]
                + "<|tool_call_argument_begin|>"
                + arguments
                + "<|tool_call_end|>"
            )
        text += "<|tool_calls_section_end|>"
    ranges["tool_call"] = (tool_call_start, len(text))
    return text, ranges


def _response_field_at(
    ranges: Dict[str, Tuple[int, int]], char_index: int
) -> str:
    for field in ("reasoning", "content", "tool_call"):
        start, end = ranges[field]
        if start <= char_index < end:
            return field
    for field in ("tool_call", "content", "reasoning"):
        start, end = ranges[field]
        if start < end == char_index:
            return field
    raise AssertionError(f"Response fork {char_index} is outside {ranges}.")


class _PandaTreeResponseSerializer:
    def apply_chat_template(
        self, messages: Sequence[Dict[str, Any]], *, tokenize: bool = False
    ) -> str:
        assert not tokenize
        response = _last_response_message(messages)
        assert response is not None
        return _serialize_response(response)[0]


def _prompt_signature(messages: Sequence[Dict[str, Any]]) -> str:
    prompt = list(messages)
    if prompt and prompt[-1].get("role") == "assistant":
        prompt = prompt[:-1]
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_annotation_time(data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "annotation_time_status": "ok",
        "annotation_time_ms": None,
        "operation_without_time_count": 0,
        "generation_without_time_count": 0,
    }
    if data.get("update_time") is None:
        result["annotation_time_status"] = "missing_update_time"
        return result

    generation_times = []
    dialogs = {
        **{str(k): v for k, v in (data.get("deleted_dialogs") or {}).items()},
        **{str(k): v for k, v in (data.get("dialogs") or {}).items()},
    }
    for dialog in dialogs.values():
        for operation in dialog.get("operations") or []:
            if operation.get("time") is None:
                result["operation_without_time_count"] += 1
            is_generation = operation.get("operator") in {
                "generate_new",
                "new_generate",
            } or operation.get("is_new_generated")
            if not is_generation:
                continue
            if operation.get("time") is None:
                result["generation_without_time_count"] += 1
            else:
                generation_times.append(int(operation["time"]))

    if not generation_times:
        result["annotation_time_status"] = "missing_first_generation_time"
        return result

    start_time = min(generation_times)
    update_time = int(data["update_time"])
    if update_time < start_time or any(time > update_time for time in generation_times):
        result["annotation_time_status"] = "invalid_time_order"
        return result

    result["annotation_time_ms"] = update_time - start_time
    return result


def is_good_dialog_key(dialogs: Dict[str, Dict[str, Any]], key: str) -> bool:
    if not dialogs:
        return False
    max_key = max(dialogs, key=int)
    annotate = dialogs[key].get("annotate") or {}
    value = annotate.get("is_good")
    if value is None:
        return key == max_key
    return bool(value)


def _get_parent_key(dialog: Dict[str, Any]) -> Optional[str]:
    operations = dialog.get("operations") or []
    if not operations:
        return None
    parent = operations[0].get("parent")
    if parent is None:
        return None
    return str(parent)


def get_parent_chain(
    dialogs: Dict[str, Dict[str, Any]], dialog_key: str
) -> Tuple[List[str], Optional[str]]:
    chain = [dialog_key]
    seen = {dialog_key}
    current = dialog_key
    while True:
        dialog = dialogs[current]
        operations = dialog.get("operations") or []
        if operations and (
            operations[0].get("is_new_generated")
            or operations[0].get("is_prompt_modified")
            or operations[0].get("operator") == "edit_prompt"
        ):
            break

        parent = _get_parent_key(dialog)
        if parent is None:
            break
        if parent not in dialogs:
            return list(reversed(chain)), "missing_parent"
        if parent in seen:
            return list(reversed(chain)), "parent_cycle"
        if _prompt_signature(
            dialogs[parent].get("messages") or []
        ) != _prompt_signature(dialog.get("messages") or []):
            return list(reversed(chain)), "prompt_changed"
        chain.append(parent)
        seen.add(parent)
        current = parent
    return list(reversed(chain)), None


def classify_operation(
    operation: Dict[str, Any],
    *,
    dialog_key: str,
    dialogs: Dict[str, Dict[str, Any]],
) -> str:
    operator = operation.get("operator")
    if operator == "continue_with_chosen":
        return "candidate"
    if (
        operator in {"continue_with_input", "edit_selection"}
        or "continue_with_input" in operation
    ):
        return "manual"
    if _is_regenerate_operation(operation, dialog_key=dialog_key, dialogs=dialogs):
        return "regenerate"
    if operator in {"generate_new", "new_generate"} or operation.get(
        "is_new_generated"
    ):
        return "new_generation"
    if operator == "continue_generating":
        return "model_continue"
    if operator in {"run_tool_calls", "start_new_round"}:
        return operator
    return str(operator or "unknown")


def _is_regenerate_operation(
    operation: Dict[str, Any],
    *,
    dialog_key: str,
    dialogs: Dict[str, Dict[str, Any]],
) -> bool:
    operator = operation.get("operator")
    if operator not in {"generate_new", "new_generate"} and not operation.get(
        "is_new_generated"
    ):
        return False
    parent_key = operation.get("parent")
    if parent_key is None:
        return False
    parent_key = str(parent_key)
    if parent_key == dialog_key or parent_key not in dialogs:
        return False
    parent_messages = dialogs[parent_key].get("messages") or []
    current_messages = dialogs[dialog_key].get("messages") or []
    if not _last_response_message(parent_messages):
        return False
    return _prompt_signature(parent_messages) == _prompt_signature(current_messages)


def _operation_fork_position(
    parent_text: str, child_text: str, token_counter: TokenCounter
) -> Tuple[int, int, int, Optional[float]]:
    common_chars = 0
    for left, right in zip(parent_text, child_text):
        if left != right:
            break
        common_chars += 1
    rejected_len = token_counter.count(parent_text)
    fork_idx = token_counter.fork_index(parent_text, common_chars)
    ratio = None
    if rejected_len > 0:
        ratio = min(max(fork_idx / rejected_len, 0.0), 1.0)
    return common_chars, fork_idx, rejected_len, ratio


def _chosen_token_info(
    operation: Dict[str, Any],
) -> Tuple[int, Optional[float], Optional[int]]:
    chosen = operation.get("continue_with_chosen") or {}
    logprob = chosen.get("logprob")
    prob = None
    if isinstance(logprob, (int, float)):
        prob = math.exp(float(logprob))

    logprob_content = (
        operation.get("rejected_token", {})
        .get("logprobs", {})
        .get("content", [])
    )
    top_items = []
    if isinstance(logprob_content, list) and logprob_content:
        nested_top_items = logprob_content[0].get("top_logprobs")
        if isinstance(nested_top_items, list):
            top_items = nested_top_items
    rank = None
    top_k = len(top_items)
    if top_items:
        chosen_token_id = chosen.get("token_id")
        chosen_token = chosen.get("token")
        for idx, item in enumerate(top_items, start=1):
            if not isinstance(item, dict):
                continue
            if (
                chosen_token_id is not None
                and item.get("token_id") == chosen_token_id
            ) or (chosen_token is not None and item.get("token") == chosen_token):
                rank = idx
                break

    return int(top_k or 0), prob, rank


def _manual_input_text(operation: Dict[str, Any]) -> Optional[str]:
    input_text = None
    for key in ("continue_with_input", "edit_selection"):
        value = operation.get(key)
        if isinstance(value, dict):
            if "input_patch" in value:
                input_text = str(value.get("input_patch") or "")
                break
            if "text" in value:
                input_text = str(value.get("text") or "")
                break
            if "content" in value:
                input_text = str(value.get("content") or "")
                break
        elif isinstance(value, str):
            input_text = value
            break
    if input_text is None and operation.get("operator") == "edit_selection":
        input_text = str(operation.get("edit_selection_text") or "")
    return "" if input_text == "<|stop|>" else input_text


def _candidate_text(operation: Dict[str, Any]) -> Optional[str]:
    value = operation.get("continue_with_chosen") or {}
    if value.get("finish_reason") is not None:
        return ""
    if value.get("bytes") is not None:
        return bytes(value["bytes"]).decode("utf-8")
    if value.get("token") is not None:
        return str(value["token"])
    return None


def _resource_ref(value: Any) -> str:
    if isinstance(value, str):
        match = re.fullmatch(onpanda.HASH_TEMPLATE_REGEX, value)
        if match:
            return match.group(1)
    return mxlm.hash_object_sha256_base64(value)


def _scan_multimodal_in_content(content: Any) -> Tuple[Counter, Dict[str, set]]:
    counts: Counter = Counter()
    uniques: Dict[str, set] = defaultdict(set)
    if not isinstance(content, list):
        return counts, uniques

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        modality = None
        ref = None
        if item_type == "image_url" or "image_url" in item:
            modality = "image"
            ref = _resource_ref(item.get("image_url"))
        elif (
            item_type in {"input_audio", "audio"}
            or "audio_path" in item
            or "input_audio" in item
        ):
            modality = "audio"
            ref = _resource_ref(
                item.get("audio_path") or item.get("input_audio") or item
            )
        elif item_type in {"video", "video_url"} or "video_url" in item:
            modality = "video"
            ref = _resource_ref(item.get("video_url") or item)

        if modality:
            counts[modality] += 1
            uniques[modality].add(ref)
    return counts, uniques


def _scan_multimodal(
    dialogs: Dict[str, Dict[str, Any]],
) -> Tuple[Counter, Dict[str, set]]:
    counts: Counter = Counter()
    uniques: Dict[str, set] = defaultdict(set)
    for dialog in dialogs.values():
        for message in dialog.get("messages") or []:
            msg_counts, msg_uniques = _scan_multimodal_in_content(
                message.get("content")
            )
            counts.update(msg_counts)
            for modality, refs in msg_uniques.items():
                uniques[modality].update(refs)
    return counts, uniques


def _tool_call_count(messages: Sequence[Dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
        elif tool_calls:
            count += 1
    return count


def statistic_one_panda_json(
    path: Path,
    *,
    token_counter: TokenCounter,
    modality: str,
    max_annotation_hours: float = DEFAULT_MAX_ANNOTATION_HOURS,
    paste_threshold: float = DEFAULT_PASTE_THRESHOLD,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    restored_message_roles = (
        _restore_agentic_message_roles(data) if modality == "agentic" else Counter()
    )
    _recover_response_hashes(data)

    sample: Dict[str, Any] = {"path": str(path), "uuid": data.get("uuid")}
    sample["restored_message_role_counts"] = dict(restored_message_roles)
    sample["label_user"] = (data.get("sado_info") or {}).get("label_user")

    dialogs: Dict[str, Dict[str, Any]] = {
        str(key): dialog for key, dialog in (data.get("dialogs") or {}).items()
    }
    deleted_dialogs = {
        str(k): v for k, v in (data.get("deleted_dialogs") or {}).items()
    }
    all_dialogs = {**deleted_dialogs, **dialogs}
    sample["dialog_count"] = len(dialogs)
    sample["deleted_dialog_count"] = len(deleted_dialogs)
    sample["incomplete_tool_call_count"] = 0
    for dialog in all_dialogs.values():
        for message in dialog.get("messages") or []:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            sample["incomplete_tool_call_count"] += sum(
                1
                for tool_call in tool_calls
                if not tool_call.get("id")
                or not (tool_call.get("function") or {}).get("name")
                or "arguments" not in (tool_call.get("function") or {})
            )

    sample.update(compute_annotation_time(data))
    sample["duration_excluded_over_limit"] = bool(
        sample["annotation_time_ms"] is not None
        and sample["annotation_time_ms"] > max_annotation_hours * HOUR_MS
    )

    multimodal_counts, multimodal_uniques = _scan_multimodal(dialogs)
    sample["multimodal_counts"] = dict(multimodal_counts)
    sample["multimodal_unique_counts"] = {
        resource_type: len(refs)
        for resource_type, refs in multimodal_uniques.items()
        if resource_type in SUPPORTED_MULTIMODAL_TYPES[modality]
    }
    sample["unsupported_multimodal_unique_counts"] = {
        resource_type: len(refs)
        for resource_type, refs in multimodal_uniques.items()
        if resource_type not in SUPPORTED_MULTIMODAL_TYPES[modality]
    }

    operation_counts: Counter = Counter()
    for dialog_key, dialog in all_dialogs.items():
        for operation in dialog.get("operations") or []:
            op_class = classify_operation(
                operation, dialog_key=dialog_key, dialogs=all_dialogs
            )
            operation_counts[op_class] += 1
    sample["operation_counts"] = dict(operation_counts)

    response_keys = [
        key
        for key, dialog in dialogs.items()
        if dialog.get("messages")
        and dialog["messages"][-1].get("role") == "assistant"
    ]
    is_good_keys = [key for key in response_keys if is_good_dialog_key(dialogs, key)]
    if is_good_keys:
        tree = onpanda.PandaTree(
            deepcopy(data),
            tokenizer=(
                _PandaTreeResponseSerializer() if modality == "agentic" else None
            ),
        )
        sample["positive_count"] = len(tree.dense_keys)
        sample["negative_count"] = len(tree.outcome_pairs) + len(tree.fork_pairs)
        sample["fork_pair_count"] = len(tree.fork_pairs)
        sample["outcome_pair_count"] = len(tree.outcome_pairs)
    else:
        sample["positive_count"] = 0
        sample["negative_count"] = 0
        sample["fork_pair_count"] = 0
        sample["outcome_pair_count"] = 0
    sample["preference_pair_count"] = sample["negative_count"]

    sample["is_good_keys"] = is_good_keys
    sample["zero_is_good_anomaly"] = len(is_good_keys) == 0

    token_origin = Counter()
    correction_counts = Counter()
    position_ratios: List[float] = []
    first_position_ratios: List[float] = []
    paste_like_edit_excluded_count = 0
    chosen_rank_values: List[int] = []
    chosen_prob_values: List[float] = []
    chosen_top_k_counter: Counter = Counter()
    user_round_counts: List[int] = []
    tool_call_counts: List[int] = []
    token_attributed_operation_counts: Counter = Counter()
    token_attribution_unavailable_reasons: Counter = Counter()
    correction_path_unavailable_reasons: Counter = Counter()
    token_origin_available_is_good_count = 0
    correction_path_available_is_good_count = 0
    mixed_source_tokens = 0

    for good_key in is_good_keys:
        good_messages = dialogs[good_key].get("messages") or []
        final_response = _last_response_message(good_messages)
        assert final_response is not None
        final_text, _ = _serialize_response(final_response)
        final_tokens = token_counter.count(final_text)

        kept_corrections: List[Dict[str, Any]] = []
        chain, parent_chain_status = get_parent_chain(all_dialogs, good_key)
        root_dialog = all_dialogs[chain[0]]
        root_response = _last_response_message(root_dialog.get("messages") or [])
        assert root_response is not None
        current_text, _ = _serialize_response(root_response)
        current_sources = ["model"] * len(current_text)
        root_corrections = [
            classify_operation(
                operation, dialog_key=chain[0], dialogs=all_dialogs
            )
            for operation in root_dialog.get("operations") or []
        ]
        root_corrections = [
            op_class
            for op_class in root_corrections
            if op_class in {"candidate", "manual"}
        ]
        correction_path_unavailable_reason = None
        if len(root_corrections) > 1:
            correction_path_unavailable_reason = "multiple_corrections_in_dialog"
        elif root_corrections:
            correction_path_unavailable_reason = (
                f"{parent_chain_status or 'missing_parent'}_for_"
                f"{root_corrections[0]}"
            )
        attribution_unavailable_reason = correction_path_unavailable_reason

        for child_key in chain[1:]:
            child_dialog = all_dialogs[child_key]
            parent_key = _get_parent_key(child_dialog)
            assert parent_key in all_dialogs
            parent_dialog = all_dialogs[parent_key]
            parent_response = _last_response_message(
                parent_dialog.get("messages") or []
            )
            child_response = _last_response_message(child_dialog.get("messages") or [])
            assert parent_response is not None and child_response is not None
            parent_text, parent_ranges = _serialize_response(parent_response)
            child_text, _ = _serialize_response(child_response)
            assert parent_text == current_text
            common_chars, fork_idx, rejected_len, ratio = _operation_fork_position(
                parent_text, child_text, token_counter
            )
            classified_operations = [
                (
                    operation,
                    classify_operation(
                        operation, dialog_key=child_key, dialogs=all_dialogs
                    ),
                )
                for operation in child_dialog.get("operations") or []
            ]
            correction_operations = [
                (operation, op_class)
                for operation, op_class in classified_operations
                if op_class in {"candidate", "manual"}
            ]
            if len(correction_operations) > 1:
                correction_path_unavailable_reason = (
                    correction_path_unavailable_reason
                    or "multiple_corrections_in_dialog"
                )
                attribution_unavailable_reason = (
                    attribution_unavailable_reason
                    or "multiple_corrections_in_dialog"
                )
                current_text = child_text
                continue

            if correction_operations:
                operation, op_class = correction_operations[0]
                patch_text = (
                    _candidate_text(operation)
                    if op_class == "candidate"
                    else _manual_input_text(operation)
                )
                patch_start = common_chars
                paste_excluded = False
                if patch_text is None:
                    attribution_unavailable_reason = f"missing_{op_class}_text"
                elif attribution_unavailable_reason is None:
                    patch_starts = [
                        start
                        for start in range(common_chars, -1, -1)
                        if child_text.startswith(patch_text, start)
                        and parent_text[:start] == child_text[:start]
                    ]
                    if not patch_starts:
                        attribution_unavailable_reason = f"{op_class}_text_not_found"
                    else:
                        patch_start = patch_starts[0]
                        source = op_class
                        if (
                            op_class == "manual"
                            and final_tokens
                            and token_counter.count(patch_text)
                            > final_tokens * paste_threshold
                        ):
                            source = "model"
                            paste_excluded = True
                        current_sources = (
                            current_sources[:patch_start]
                            + [source] * len(patch_text)
                            + ["model"]
                            * (len(child_text) - patch_start - len(patch_text))
                        )

                fork_idx = token_counter.fork_index(parent_text, patch_start)
                ratio = fork_idx / rejected_len if rejected_len else None
                while (
                    kept_corrections
                    and kept_corrections[-1]["fork_char_idx"] >= patch_start
                ):
                    kept_corrections.pop()
                kept_corrections.append(
                    {
                        "operation": operation,
                        "class": op_class,
                        "fork_char_idx": patch_start,
                        "fork_idx": fork_idx,
                        "rejected_len": rejected_len,
                        "ratio": ratio,
                        "field": (
                            "content"
                            if not parent_text and modality in {"vlm", "audio"}
                            else _response_field_at(parent_ranges, patch_start)
                        ),
                        "parent_finish_reason": parent_response.get("finish_reason"),
                        "paste_excluded": paste_excluded,
                    }
                )
            else:
                if any(
                    op_class == "regenerate"
                    for _, op_class in classified_operations
                ):
                    kept_corrections.clear()
                    if attribution_unavailable_reason is None:
                        current_sources = ["model"] * len(child_text)
                else:
                    while (
                        kept_corrections
                        and kept_corrections[-1]["fork_char_idx"] >= common_chars
                    ):
                        kept_corrections.pop()
                    if attribution_unavailable_reason is None:
                        current_sources = (
                            current_sources[:common_chars]
                            + ["model"] * (len(child_text) - common_chars)
                        )
            current_text = child_text

        if correction_path_unavailable_reason is None:
            correction_path_available_is_good_count += 1
            for correction in kept_corrections:
                operation = correction["operation"]
                op_class = correction["class"]
                if op_class == "candidate":
                    top_k, prob, rank = _chosen_token_info(operation)
                    if top_k:
                        chosen_top_k_counter[str(top_k)] += 1
                    if prob is not None:
                        chosen_prob_values.append(prob)
                    if rank is not None:
                        chosen_rank_values.append(rank)
                if correction["paste_excluded"]:
                    paste_like_edit_excluded_count += 1

                correction_counts[op_class] += 1
                correction_counts[correction["field"]] += 1
                if (
                    correction["ratio"] is not None
                    and correction["parent_finish_reason"]
                    in COMPLETE_ROLLOUT_FINISH_REASONS
                ):
                    position_ratios.append(correction["ratio"])

            if kept_corrections:
                first = kept_corrections[0]
                if (
                    first["ratio"] is not None
                    and first["parent_finish_reason"]
                    in COMPLETE_ROLLOUT_FINISH_REASONS
                ):
                    first_position_ratios.append(first["ratio"])
            else:
                correction_counts["direct_accept"] += 1
        else:
            correction_path_unavailable_reasons[
                correction_path_unavailable_reason
            ] += 1

        assert current_text == final_text
        if attribution_unavailable_reason is None:
            source_counts, mixed_tokens = token_counter.count_sources(
                final_text, current_sources
            )
            assert sum(source_counts.values()) == final_tokens
            token_origin["final_total_tokens"] += final_tokens
            token_origin["model_generated_tokens"] += source_counts["model"]
            token_origin["candidate_selected_tokens"] += source_counts["candidate"]
            token_origin["annotator_typed_tokens"] += source_counts["manual"]
            token_attributed_operation_counts.update(
                correction["class"] for correction in kept_corrections
            )
            token_origin_available_is_good_count += 1
            mixed_source_tokens += mixed_tokens
        else:
            token_attribution_unavailable_reasons[attribution_unavailable_reason] += 1
        user_round_counts.append(
            sum(1 for message in good_messages if message.get("role") == "user")
        )
        tool_call_counts.append(_tool_call_count(good_messages))

    sample["is_good_count_for_averages"] = len(is_good_keys)
    sample["token_origin"] = dict(token_origin)
    sample["correction_counts"] = dict(correction_counts)
    sample["position_ratios"] = position_ratios
    sample["first_position_ratios"] = first_position_ratios
    sample["paste_like_edit_excluded_count"] = paste_like_edit_excluded_count
    sample["chosen_token_rank_values"] = chosen_rank_values
    sample["chosen_token_prob_values"] = chosen_prob_values
    sample["chosen_token_top_k_counter"] = dict(chosen_top_k_counter)
    sample["token_attributed_operation_counts"] = dict(
        token_attributed_operation_counts
    )
    sample["token_attribution_unavailable_reasons"] = dict(
        token_attribution_unavailable_reasons
    )
    sample["correction_path_unavailable_reasons"] = dict(
        correction_path_unavailable_reasons
    )
    sample["correction_path_available_is_good_count"] = (
        correction_path_available_is_good_count
    )
    sample["token_origin_available_is_good_count"] = (
        token_origin_available_is_good_count
    )
    sample["mixed_source_tokens"] = mixed_source_tokens
    sample["user_round_counts"] = user_round_counts
    sample["tool_call_counts"] = tool_call_counts
    sample["max_user_rounds"] = max(user_round_counts) if user_round_counts else 0
    sample["max_tool_calls"] = max(tool_call_counts) if tool_call_counts else 0
    return sample


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if not denominator:
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * percentile
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[int(idx)]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def _distribution_5_bins(values: Sequence[float]) -> Dict[str, float]:
    bins = {
        "[0,20%)": 0,
        "[20,40%)": 0,
        "[40,60%)": 0,
        "[60,80%)": 0,
        "[80,100%]": 0,
    }
    if not values:
        return {key: 0.0 for key in bins}
    for value in values:
        if value < 0.2:
            bins["[0,20%)"] += 1
        elif value < 0.4:
            bins["[20,40%)"] += 1
        elif value < 0.6:
            bins["[40,60%)"] += 1
        elif value < 0.8:
            bins["[60,80%)"] += 1
        else:
            bins["[80,100%]"] += 1
    total = len(values)
    return {key: count / total for key, count in bins.items()}


def aggregate_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    tokenizer_name: str,
    modality: str,
    max_annotation_hours: float,
    paste_threshold: float,
) -> Dict[str, Any]:
    session_count = len(samples)

    duration_values = [
        sample["annotation_time_ms"]
        for sample in samples
        if sample.get("annotation_time_ms") is not None
        and not sample.get("duration_excluded_over_limit")
    ]
    duration_unavailable = Counter(
        sample.get("annotation_time_status")
        for sample in samples
        if sample.get("annotation_time_status") != "ok"
    )
    duration_excluded_7h = sum(
        1 for sample in samples if sample.get("duration_excluded_over_limit")
    )

    label_users = {
        sample.get("label_user")
        for sample in samples
        if sample.get("label_user")
    }
    label_user_available = sum(1 for sample in samples if sample.get("label_user"))

    total_positive = sum(sample.get("positive_count", 0) for sample in samples)
    total_negative = sum(sample.get("negative_count", 0) for sample in samples)
    total_preference_pairs = sum(
        sample.get("preference_pair_count", 0) for sample in samples
    )
    total_is_good_for_avg = sum(
        sample.get("is_good_count_for_averages", 0) for sample in samples
    )

    token_origin = Counter()
    correction_counts = Counter()
    operation_counts = Counter()
    multimodal_unique_counts = Counter()
    unsupported_multimodal_unique_counts = Counter()
    restored_message_role_counts = Counter()
    token_attributed_operation_counts = Counter()
    token_attribution_unavailable_reasons = Counter()
    correction_path_unavailable_reasons = Counter()
    position_ratios: List[float] = []
    first_position_ratios: List[float] = []
    chosen_rank_values: List[int] = []
    chosen_prob_values: List[float] = []
    chosen_top_k_counter: Counter = Counter()
    token_origin_available_is_good_count = 0
    correction_path_available_is_good_count = 0
    mixed_source_tokens = 0

    for sample in samples:
        token_origin.update(sample.get("token_origin") or {})
        correction_counts.update(sample.get("correction_counts") or {})
        operation_counts.update(sample.get("operation_counts") or {})
        multimodal_unique_counts.update(sample.get("multimodal_unique_counts") or {})
        unsupported_multimodal_unique_counts.update(
            sample.get("unsupported_multimodal_unique_counts") or {}
        )
        restored_message_role_counts.update(
            sample.get("restored_message_role_counts") or {}
        )
        token_attributed_operation_counts.update(
            sample.get("token_attributed_operation_counts") or {}
        )
        token_attribution_unavailable_reasons.update(
            sample.get("token_attribution_unavailable_reasons") or {}
        )
        correction_path_unavailable_reasons.update(
            sample.get("correction_path_unavailable_reasons") or {}
        )
        position_ratios.extend(sample.get("position_ratios") or [])
        first_position_ratios.extend(sample.get("first_position_ratios") or [])
        chosen_rank_values.extend(sample.get("chosen_token_rank_values") or [])
        chosen_prob_values.extend(sample.get("chosen_token_prob_values") or [])
        chosen_top_k_counter.update(sample.get("chosen_token_top_k_counter") or {})
        token_origin_available_is_good_count += sample.get(
            "token_origin_available_is_good_count", 0
        )
        correction_path_available_is_good_count += sample.get(
            "correction_path_available_is_good_count", 0
        )
        mixed_source_tokens += sample.get("mixed_source_tokens", 0)

    three_operation_total = (
        operation_counts.get("regenerate", 0)
        + operation_counts.get("candidate", 0)
        + operation_counts.get("manual", 0)
    )
    token_level_correction_count = operation_counts.get(
        "candidate", 0
    ) + operation_counts.get("manual", 0)

    summary = {
        "config": {
            "tokenizer": tokenizer_name,
            "modality": modality,
            "max_annotation_hours": max_annotation_hours,
            "paste_threshold": paste_threshold,
        },
        "panda_json_count": len(samples),
        "restored_message_role_counts": dict(restored_message_role_counts),
        "incomplete_tool_call_count": sum(
            sample.get("incomplete_tool_call_count", 0)
            for sample in samples
        ),
        "project_annotator_count": len(label_users),
        "label_user_available_count": label_user_available,
        "label_user_coverage": _safe_div(label_user_available, session_count),
        "annotation_time": {
            "p25_ms": _percentile(duration_values, 0.25),
            "p50_ms": _percentile(duration_values, 0.50),
            "p75_ms": _percentile(duration_values, 0.75),
            "available_count": len(duration_values),
            "excluded_over_7h_count": duration_excluded_7h,
            "unavailable_reasons": dict(duration_unavailable),
            "operation_without_time_count": sum(
                sample.get("operation_without_time_count", 0)
                for sample in samples
            ),
            "generation_without_time_count": sum(
                sample.get("generation_without_time_count", 0)
                for sample in samples
            ),
            "total_elapsed_hours": (
                sum(duration_values) / HOUR_MS if duration_values else 0.0
            ),
            "total_person_month_equivalents": (
                sum(duration_values) / HOUR_MS / 160 if duration_values else 0.0
            ),
        },
        "positive_count": total_positive,
        "negative_count": total_negative,
        "preference_pair_count": total_preference_pairs,
        "positive_avg_per_session": _safe_div(total_positive, session_count),
        "negative_avg_per_session": _safe_div(total_negative, session_count),
        "token_level_correction_count": token_level_correction_count,
        "token_level_corrections_avg_per_session": _safe_div(
            token_level_correction_count, session_count
        ),
        "token_origin_avg_per_is_good": {
            key: _safe_div(
                token_origin.get(key, 0), token_origin_available_is_good_count
            )
            for key in (
                "final_total_tokens",
                "model_generated_tokens",
                "candidate_selected_tokens",
                "annotator_typed_tokens",
            )
        },
        "token_origin_available_is_good_count": token_origin_available_is_good_count,
        "token_origin_coverage": _safe_div(
            token_origin_available_is_good_count, total_is_good_for_avg
        ),
        "token_attribution_unavailable_reasons": dict(
            token_attribution_unavailable_reasons
        ),
        "token_attributed_operation_counts": dict(
            token_attributed_operation_counts
        ),
        "token_avg_per_attributed_operation": {
            "candidate": _safe_div(
                token_origin.get("candidate_selected_tokens", 0),
                token_attributed_operation_counts.get("candidate", 0),
            ),
            "manual": _safe_div(
                token_origin.get("annotator_typed_tokens", 0),
                token_attributed_operation_counts.get("manual", 0),
            ),
        },
        "mixed_source_tokens": mixed_source_tokens,
        "is_good_count_for_averages": total_is_good_for_avg,
        "correction_path_available_is_good_count": (
            correction_path_available_is_good_count
        ),
        "correction_path_coverage": _safe_div(
            correction_path_available_is_good_count, total_is_good_for_avg
        ),
        "correction_path_unavailable_reasons": dict(
            correction_path_unavailable_reasons
        ),
        "correction_avg_per_is_good": {
            **{
                field: _safe_div(
                    correction_counts.get(field, 0),
                    correction_path_available_is_good_count,
                )
                for field in ("reasoning", "content", "tool_call")
            },
            "total": _safe_div(
                sum(
                    correction_counts.get(field, 0)
                    for field in ("reasoning", "content", "tool_call")
                ),
                correction_path_available_is_good_count,
            ),
            "direct_accept_rate": _safe_div(
                correction_counts.get("direct_accept", 0),
                correction_path_available_is_good_count,
            ),
        },
        "position_distribution": {
            "mean": py_statistics.mean(position_ratios) if position_ratios else None,
            "count": len(position_ratios),
            "bins": _distribution_5_bins(position_ratios),
            "first_token_modification_rate": _safe_div(
                sum(1 for v in position_ratios if v == 0.0), len(position_ratios)
            ),
        },
        "first_correction_position_distribution": {
            "mean": py_statistics.mean(first_position_ratios)
            if first_position_ratios
            else None,
            "count": len(first_position_ratios),
            "bins": _distribution_5_bins(first_position_ratios),
        },
        "operation_counts": dict(operation_counts),
        "operation_avg_per_session": {
            operation: _safe_div(operation_counts.get(operation, 0), session_count)
            for operation in ("candidate", "manual", "regenerate")
        },
        "final_path_operation_counts": {
            "candidate": correction_counts.get("candidate", 0),
            "manual": correction_counts.get("manual", 0),
        },
        "operation_share": {
            "regenerate": _safe_div(
                operation_counts.get("regenerate", 0), three_operation_total
            ),
            "candidate": _safe_div(
                operation_counts.get("candidate", 0), three_operation_total
            ),
            "manual": _safe_div(
                operation_counts.get("manual", 0), three_operation_total
            ),
            "three_operation_total": three_operation_total,
        },
        "tool_call": {
            "avg_session_max_in_is_good": py_statistics.mean(
                sample.get("max_tool_calls", 0) for sample in samples
            )
            if samples
            else None,
        },
        "user_round": {
            "avg_session_max_in_is_good": py_statistics.mean(
                sample.get("max_user_rounds", 0) for sample in samples
            )
            if samples
            else None,
        },
        "multimodal": {
            "avg_unique_count_per_session": {
                resource_type: _safe_div(count, session_count)
                for resource_type, count in sorted(multimodal_unique_counts.items())
            },
            "unsupported_unique_count": dict(
                sorted(unsupported_multimodal_unique_counts.items())
            ),
        },
        "chosen_token": {
            "rank_available_count": len(chosen_rank_values),
            "rank_mean": py_statistics.mean(chosen_rank_values)
            if chosen_rank_values
            else None,
            "prob_available_count": len(chosen_prob_values),
            "prob_mean": py_statistics.mean(chosen_prob_values)
            if chosen_prob_values
            else None,
            "top_k_counter": dict(chosen_top_k_counter),
        },
        "paste_like_edit_excluded_count": sum(
            sample.get("paste_like_edit_excluded_count", 0) for sample in samples
        ),
        "zero_is_good_anomaly_count": sum(
            1 for sample in samples if sample.get("zero_is_good_anomaly")
        ),
    }
    return summary


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def _format_minutes(ms: Optional[float]) -> str:
    if ms is None:
        return "N/A"
    return f"{ms / 60000:.2f} min"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_metric_rows(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    modality = summary["config"]["modality"]
    ann = summary["annotation_time"]
    token_avg = summary["token_origin_avg_per_is_good"]
    token_operation_avg = summary["token_avg_per_attributed_operation"]
    correction_avg = summary["correction_avg_per_is_good"]
    position = summary["position_distribution"]
    first_position = summary["first_correction_position_distribution"]
    operation_share = summary["operation_share"]
    operation_counts = summary["operation_counts"]
    operation_avg = summary["operation_avg_per_session"]
    final_path_counts = summary["final_path_operation_counts"]
    multimodal = summary["multimodal"]
    chosen = summary["chosen_token"]
    other_operation_counts = {
        operation: count
        for operation, count in operation_counts.items()
        if operation not in {"candidate", "manual", "regenerate"}
    }

    rows = [
        {
            "metric_name": "Annotation sessions",
            "value": str(summary["panda_json_count"]),
            "note": (
                f"0 条 is_good 异常 {summary['zero_is_good_anomaly_count']}；"
                f"补全缺失 message role "
                f"{sum(summary['restored_message_role_counts'].values())}；"
                f"不完整 tool_call {summary['incomplete_tool_call_count']}"
            ),
        },
        {
            "metric_name": "Annotators",
            "value": str(summary["project_annotator_count"]),
            "note": (
                "按当前导出中实际出现的 sado_info.label_user 去重；"
                f"覆盖率 {_format_percent(summary['label_user_coverage'])}"
            ),
        },
        {
            "metric_name": "Annotation time P25 / P50 / P75 (min/session)",
            "value": (
                f"{_format_minutes(ann['p25_ms'])} / "
                f"{_format_minutes(ann['p50_ms'])} / "
                f"{_format_minutes(ann['p75_ms'])}"
            ),
            "note": (
                f"可用 {ann['available_count']}，"
                f">7h 剔除 {ann['excluded_over_7h_count']}，"
                f"时间不可用 {sum(ann['unavailable_reasons'].values())}；"
                f"起点为首次 gen，"
                f"缺时间 gen {ann['generation_without_time_count']}；"
                f"总 {_format_number(ann['total_elapsed_hours'], 2)} elapsed-hours / "
                f"{_format_number(ann['total_person_month_equivalents'], 2)} "
                f"person-month equivalents（160h/月）"
            ),
        },
        {
            "metric_name": "SFT samples / negative samples (/session)",
            "value": (
                f"{_format_number(summary['positive_avg_per_session'])} / "
                f"{_format_number(summary['negative_avg_per_session'])}"
            ),
            "note": (
                f"总数 {summary['positive_count']} / {summary['negative_count']}；"
                f"is_good=None 按最新 dialog 为正样本"
            ),
        },
        {
            "metric_name": "Token-level corrections (/session)",
            "value": _format_number(
                summary["token_level_corrections_avg_per_session"]
            ),
            "note": (
                f"候选点击 + 双击修改总数 "
                f"{summary['token_level_correction_count']}"
            ),
        },
        {
            "metric_name": (
                "平均每条 is_good 样本中最后 response 中 "
                "模型生成/候选续写/标注员输入"
                "（双击修改）的 tokens 数"
            ),
            "value": (
                f"{_format_number(token_avg['model_generated_tokens'])} / "
                f"{_format_number(token_avg['candidate_selected_tokens'])} / "
                f"{_format_number(token_avg['annotator_typed_tokens'])}"
            ),
            "note": (
                "Qwen2.5 tokenizer；可归因 "
                f"{summary['token_origin_available_is_good_count']} / "
                f"{summary['is_good_count_for_averages']} "
                f"({_format_percent(summary['token_origin_coverage'])})；"
                "每次候选/双击 "
                f"{_format_number(token_operation_avg['candidate'])} / "
                f"{_format_number(token_operation_avg['manual'])} "
                "tokens；"
                f"paste-like edit 剔除 {summary['paste_like_edit_excluded_count']}"
            ),
        },
        {
            "metric_name": (
                "平均每条 is_good 的 reasoning/content/tool_call 修改次数"
            ),
            "value": (
                f"{_format_number(correction_avg['reasoning'])} / "
                f"{_format_number(correction_avg['content'])} / "
                f"{_format_number(correction_avg['tool_call'])}"
            ),
            "note": (
                (
                    "完整 response 按 reasoning/content/tool_call 定位；"
                    if modality == "agentic"
                    else f"{modality.upper()} 只统计 content；"
                )
                + "路径覆盖 "
                f"{summary['correction_path_available_is_good_count']} / "
                f"{summary['is_good_count_for_averages']} "
                f"({_format_percent(summary['correction_path_coverage'])})；"
                f"0 修正直接合格率 "
                f"{_format_percent(correction_avg['direct_accept_rate'])}"
            ),
        },
        {
            "metric_name": "百分比位置分布",
            "value": _format_number(position["mean"]),
            "note": (
                f"count {position['count']}；bins "
                f"{json.dumps(position['bins'], ensure_ascii=False)}；"
                "首 token 修改率 "
                f"{_format_percent(position['first_token_modification_rate'])}；"
                f"首次修正均值 {_format_number(first_position['mean'])}，"
                f"count {first_position['count']}"
            ),
        },
        {
            "metric_name": "Operations (click / edit / regen.) (/session)",
            "value": (
                f"{_format_number(operation_avg['candidate'])} / "
                f"{_format_number(operation_avg['manual'])} / "
                f"{_format_number(operation_avg['regenerate'])}"
            ),
            "note": (
                f"总数 click/edit/regen "
                f"{operation_counts.get('candidate', 0)} / "
                f"{operation_counts.get('manual', 0)} / "
                f"{operation_counts.get('regenerate', 0)}；"
                f"占比 {_format_percent(operation_share['candidate'])} / "
                f"{_format_percent(operation_share['manual'])} / "
                f"{_format_percent(operation_share['regenerate'])}；"
                "最终 is_good 路径候选/双击 "
                f"{final_path_counts['candidate']} / "
                f"{final_path_counts['manual']}；"
                f"其它操作 "
                f"{json.dumps(other_operation_counts, ensure_ascii=False)}"
            ),
        },
        {
            "metric_name": "tool calls / user turns (/session)",
            "value": (
                f"{_format_number(summary['tool_call']['avg_session_max_in_is_good'])} "
                "/ "
                f"{_format_number(summary['user_round']['avg_session_max_in_is_good'])}"
            ),
            "note": (
                "每个 session 先取其 is_good 中最大累计值，"
                "再对全部 session 求平均"
            ),
        },
        {
            "metric_name": "Multimodal inputs (/session)",
            "value": json.dumps(
                multimodal["avg_unique_count_per_session"], ensure_ascii=False
            ),
            "note": (
                "每个 session 内按资源内容 hash 去重"
                "（包括 tool response）；"
                "不支持类型仅报告不计入主值 "
                + json.dumps(
                    multimodal["unsupported_unique_count"], ensure_ascii=False
                )
            ),
        },
        {
            "metric_name": "chosen token 的 rank/概率分布/top-k 选择分布",
            "value": (
                f"rank_mean {_format_number(chosen['rank_mean'])}; "
                f"prob_mean {_format_number(chosen['prob_mean'])}"
            ),
            "note": (
                f"rank 覆盖 {chosen['rank_available_count']}；"
                f"prob 覆盖 {chosen['prob_available_count']}；"
                f"top-k {json.dumps(chosen['top_k_counter'], ensure_ascii=False)}"
            ),
        },
    ]
    return rows


def write_outputs(
    *,
    out_dir: Path,
    summary: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_metric_rows(summary)

    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    samples_path = out_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric_name", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {summary['config']['modality'].upper()} Statistics Summary\n\n")
        f.write(f"- tokenizer: `{summary['config']['tokenizer']}`\n")
        f.write(f"- panda json count: `{summary['panda_json_count']}`\n\n")
        f.write("| metric_name | value | note |\n")
        f.write("| --- | --- | --- |\n")
        for row in rows:
            metric = row["metric_name"].replace("|", "\\|")
            value = row["value"].replace("|", "\\|")
            note = row["note"].replace("|", "\\|")
            f.write(f"| {metric} | {value} | {note} |\n")

    return {
        "summary_json": str(summary_json_path),
        "samples_jsonl": str(samples_path),
        "summary_csv": str(csv_path),
        "summary_md": str(md_path),
    }


def statistic_panda_jsons(
    paths: Sequence[Path],
    *,
    modality: str = "vlm",
    tokenizer_name_or_path: str = DEFAULT_TOKENIZER,
    max_annotation_hours: float = DEFAULT_MAX_ANNOTATION_HOURS,
    paste_threshold: float = DEFAULT_PASTE_THRESHOLD,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tokenizer = load_tokenizer(tokenizer_name_or_path)
    token_counter = TokenCounter(tokenizer)
    samples: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths, start=1):
        if idx == 1 or idx % 500 == 0:
            print(f"[statistics] processing {idx}/{len(paths)}: {path}")
        samples.append(
            statistic_one_panda_json(
                path,
                token_counter=token_counter,
                modality=modality,
                max_annotation_hours=max_annotation_hours,
                paste_threshold=paste_threshold,
            )
        )

    tokenizer_name = getattr(tokenizer, "name_or_path", tokenizer_name_or_path)
    summary = aggregate_samples(
        samples,
        tokenizer_name=tokenizer_name,
        modality=modality,
        max_annotation_hours=max_annotation_hours,
        paste_threshold=paste_threshold,
    )
    return summary, samples


def merge_shard_outputs(out_dir: Path, run_id: str) -> Dict[str, str]:
    shard_dirs = sorted((out_dir / "shards" / run_id).glob("*"))
    if not shard_dirs:
        raise RuntimeError(f"No shards found for run {run_id!r} in {out_dir}.")

    shard_summaries = []
    samples = []
    for shard_dir in shard_dirs:
        with open(shard_dir / "summary.json", "r", encoding="utf-8") as f:
            shard_summary = json.load(f)
        shard_summaries.append(shard_summary)

        shard_samples = []
        with open(
            shard_dir / "samples.jsonl", "r", encoding="utf-8"
        ) as f:
            for line in f:
                shard_samples.append(json.loads(line))
        assert len(shard_samples) == shard_summary["panda_json_count"]
        samples.extend(shard_samples)

    shard_counts = {summary["shard"]["count"] for summary in shard_summaries}
    assert len(shard_counts) == 1
    shard_count = shard_counts.pop()
    shard_indexes = {summary["shard"]["index"] for summary in shard_summaries}
    assert shard_indexes == set(range(shard_count))
    assert len({sample["path"] for sample in samples}) == len(samples)

    config = shard_summaries[0]["config"]
    assert all(summary["config"] == config for summary in shard_summaries)
    summary = aggregate_samples(
        samples,
        tokenizer_name=config["tokenizer"],
        modality=config["modality"],
        max_annotation_hours=config["max_annotation_hours"],
        paste_threshold=config["paste_threshold"],
    )
    summary["parallel_run"] = {"run_id": run_id, "shard_count": shard_count}
    return write_outputs(out_dir=out_dir, summary=summary, samples=samples)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute multimodal panda-json statistics."
    )
    parser.add_argument("--config", help="Optional JSON config for one-command runs.")
    parser.add_argument(
        "sources",
        nargs="*",
        help="Input .panda.json file, directory, or glob. Quote globs containing **.",
    )
    parser.add_argument(
        "--modality", default=None, choices=["vlm", "audio", "agentic"]
    )
    parser.add_argument(
        "--shard-run-id",
        default=None,
        help="Write this rlaunch replica under outputs/shards/RUN_ID.",
    )
    parser.add_argument(
        "--merge-shards",
        default=None,
        metavar="RUN_ID",
        help="Validate and merge all shards for RUN_ID.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer name/path. Defaults to Qwen2.5; if the canonical tokenizer "
            "is not cached, Qwen2.5 GPTQ tokenizer is used as fallback."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for summary.md/csv/json and samples.jsonl.",
    )
    parser.add_argument(
        "--max-annotation-hours",
        type=float,
        default=None,
        help="Exclude elapsed annotation durations longer than this threshold.",
    )
    parser.add_argument(
        "--paste-threshold",
        type=float,
        default=None,
        help=(
            "Exclude manual input token count when it exceeds this fraction "
            "of final response tokens."
        ),
    )

    args = parser.parse_args(argv)
    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    sources = args.sources or config.get("sources", [])
    modality = args.modality or config.get("modality", "vlm")
    tokenizer = args.tokenizer or config.get("tokenizer", DEFAULT_TOKENIZER)
    out_dir = Path(
        args.out
        or config.get("out")
        or DEFAULT_OUTPUT_ROOT / f"{modality}_statistics"
    )
    max_annotation_hours = (
        args.max_annotation_hours
        if args.max_annotation_hours is not None
        else config.get("max_annotation_hours", DEFAULT_MAX_ANNOTATION_HOURS)
    )
    paste_threshold = (
        args.paste_threshold
        if args.paste_threshold is not None
        else config.get("paste_threshold", DEFAULT_PASTE_THRESHOLD)
    )

    if args.merge_shards:
        outputs = merge_shard_outputs(out_dir, args.merge_shards)
        print("\nMerged:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")
        return 0

    paths = resolve_paths(sources)
    if not paths:
        raise SystemExit("No input panda json files found.")

    replica_index = int(
        os.environ.get("RJOB_REPLICA", os.environ.get("RLAUNCH_REPLICA", 0))
    )
    replica_count = int(
        os.environ.get(
            "RJOB_REPLICA_TOTAL", os.environ.get("RLAUNCH_REPLICA_TOTAL", 1)
        )
    )
    assert 0 <= replica_index < replica_count
    if replica_count > 1:
        if not args.shard_run_id:
            raise SystemExit("--shard-run-id is required for rlaunch replicas.")
        paths = paths[replica_index::replica_count]
        out_dir = out_dir / "shards" / args.shard_run_id / str(replica_index)
        print(
            f"[statistics] shard {replica_index}/{replica_count}: "
            f"{len(paths)} panda json files"
        )

    summary, samples = statistic_panda_jsons(
        paths,
        modality=modality,
        tokenizer_name_or_path=tokenizer,
        max_annotation_hours=max_annotation_hours,
        paste_threshold=paste_threshold,
    )
    if replica_count > 1:
        summary["shard"] = {"index": replica_index, "count": replica_count}
    outputs = write_outputs(out_dir=out_dir, summary=summary, samples=samples)
    print("\nSaved:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
