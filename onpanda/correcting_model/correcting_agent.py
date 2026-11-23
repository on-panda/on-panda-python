"""Iterative tool-based correction, with policy candidates recorded as Panda dialogs."""

import json
import time
from copy import deepcopy
from uuid import uuid4

import mximport
import mxlm

with mximport.inpkg():
    from .best_of_n_mixin import BestOfNMixin
    from .correction_call.trajectory import (
        apply_correct_once,
        build_initial_cm_traj,
        policy_prompt_length,
        render_dialog,
        response_template,
    )
    from .far_correction_utils import FindAndReplaceCorrectionAdapter


def _panda_dialog(messages, tools, operation, policy_chat):
    if operation:
        operation.update(
            time=time.time_ns() // 1_000_000,
            on_policy=True,
            chat_config={
                k: v
                for k, v in policy_chat.default_kwargs.items()
                if k in ("model", "temperature", "top_p", "max_tokens")
            },
        )
    return dict(
        messages=deepcopy(messages),
        tools=deepcopy(tools),
        operations=[operation] if operation else [],
        annotate=dict(is_good=False),
    )


def _generate_new_dialog(messages, tools, policy_chat):
    rollout = policy_chat(dict(messages=messages, tools=tools), return_traj=True)
    return _panda_dialog(
        rollout["trajectory"]["messages"],
        tools,
        dict(operator="generate_new", is_new_generated=True, message_index=-1),
        policy_chat,
    )


class CorrectingAgent:
    """One CM conversation per iterative_correction call, shared across all policy dialogs."""

    def __init__(
        self,
        cm_chat,
        max_replacement_tokens,
        correction_scope="last_assistant",
        base_cm_traj=None,
        tool_name_to_call=None,
        max_correction_attempts=5,
    ):
        assert max_replacement_tokens >= 1, "max_replacement_tokens must be at least 1"
        self.chat_correcting = cm_chat
        self.max_replacement_tokens = max_replacement_tokens
        self.correction_scope = correction_scope
        self.base_cm_traj = base_cm_traj
        self.tool_name_to_call = tool_name_to_call
        self.max_correction_attempts = max_correction_attempts

    def correct(self, correcting_session, dialogs, tool_content=None):
        """Request one correct_once decision and log it on the dialog it evaluates."""
        result = correcting_session.run_agentic_loop(
            new_messages=(
                None
                if tool_content is None
                else [dict(role="tool", content=tool_content)]
            )
        )
        calls = correcting_session.pending_tool_calls
        assert len(calls) == 1 and calls[0]["function"]["name"] == "correct_once", (
            "Expected one correct_once call",
            calls,
        )
        key = max(dialogs)
        arguments_text = calls[0]["function"]["arguments"]
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError as error:
            arguments = arguments_text
            feedback = f"Invalid correct_once arguments JSON: {error}"
        else:
            key = arguments.get("dialog_key", key)
            feedback = None
            if type(key) is not int or key not in dialogs:
                feedback = (
                    f"Unknown dialog_key={key!r}; available keys: {list(dialogs)}"
                )
                key = None
            arguments_text = json.dumps(arguments, ensure_ascii=False, indent=2)

        if feedback:
            applied = dict(
                messages_location=dict(
                    not_found=True, match_num=0, find_feedback=feedback
                )
            )
        else:
            applied = (
                dict(messages_location=dict(is_good=True))
                if arguments["is_good"] is True
                else apply_correct_once(dialogs[key], arguments)
            )
            dialogs[key]["annotate"]["is_good"] = arguments["is_good"]
        record = dict(
            arguments=arguments,
            messages_location=applied["messages_location"],
            responses=result["responses"],
        )
        if key is not None:
            dialog = dialogs[key]
            dialog.setdefault("agent_corrections", []).append(record)
            dialog["annotate"].setdefault("customs", []).append(
                dict(name="correction_once", type="text", text=arguments_text)
            )
        return dict(
            dialog_key=key,
            correction=record,
            partial_messages=applied.get("partial_messages"),
        )

    def correct_and_rollout(
        self,
        correcting_session,
        dialogs,
        policy_chat,
        adapter_policy,
        tool_content=None,
        *,
        allow_rollout=True,
    ):
        """Review once (retry a missed location) and create at most one policy candidate."""
        for _ in range(self.max_correction_attempts):
            decision = self.correct(correcting_session, dialogs, tool_content)
            key, record = decision["dialog_key"], decision["correction"]
            location = record["messages_location"]
            if not location.get("not_found") or (not allow_rollout and key is not None):
                break
            tool_content = f"{location['find_feedback']}。请重新提交 correct_once。"

        if key is None:
            key = max(dialogs)
        parent = dialogs[key]
        if location.get("is_good") or not allow_rollout:
            return dict(
                dialog_key=key,
                corrected_messages=parent["messages"],
                is_good=bool(location.get("is_good")),
                tool_content=None,
            )
        tools = parent["tools"]
        if location.get("not_found"):
            fork = -1
            child_dialog = _generate_new_dialog(
                parent["messages"][:-1], tools, policy_chat
            )
            note = "定位失败，policy 已重新生成本轮 assistant。"
        else:
            fork = location["path_keys"][0]
            truncated = adapter_policy.truncate_partial_messages(
                parent["messages"],
                decision["partial_messages"],
                tools=tools,
            )
            messages = truncated["partial_messages"]
            record["truncated_continue_prefix"] = response_template.apply(messages[-1])[
                "templated_prompt"
            ]
            note = f"max_replacement_tokens={self.max_replacement_tokens}"
            if messages[-1].get("finish_reason") not in ("stop", "tool_calls"):
                messages = policy_chat(
                    dict(messages=messages, tools=tools),
                    return_traj=True,
                    continue_final_message=True,
                    skip_special_tokens=False,
                )["trajectory"]["messages"]
            operation = dict(
                operator="continue_with_input",
                is_prompt_modified=fork < policy_prompt_length(parent["messages"]),
                continue_with_input=dict(input_patch=truncated["replacement"]),
            )
            child_dialog = _panda_dialog(messages, tools, operation, policy_chat)
        child_dialog["operations"][0]["parent"] = str(key)
        child = max(dialogs) + 1
        dialogs[child] = child_dialog
        record["corrected_dialog_key"] = child
        ancestors = [key]
        while dialogs[key]["operations"] and "parent" in dialogs[key]["operations"][0]:
            key = int(dialogs[key]["operations"][0]["parent"])
            ancestors.append(key)
        (content,) = render_dialog(
            dialogs[child],
            child,
            ancestors[::-1],
            self.correction_scope,
            rejected_message=parent["messages"][fork],
        )
        return dict(
            dialog_key=child,
            corrected_messages=dialogs[child]["messages"],
            tool_content=f"<correcting_prompt>\n{note}\n</correcting_prompt>\n"
            + content["text"],
        )

    def iterative_correction_till_good(
        self, policy_session, extra_context=None, max_rollouts=5, iid_sampling=False
    ):
        """Standalone till_good entry point with the same per-call CM history."""
        result = self.iterative_correction(
            policy_session,
            extra_context,
            rollout_num=max_rollouts,
            iid_sampling=iid_sampling,
        )
        return {
            ("max_rollouts" if k == "rollout_num" else k): v for k, v in result.items()
        }

    def iterative_correction(
        self,
        policy_session,
        extra_context=None,
        rollout_num=5,
        mode="till_good",
        iid_sampling=False,
    ):
        """Review policy dialogs in one CM conversation without changing the policy session.

        A good verdict ends the current round. Multi-candidate modes sample the next
        root from the original policy prompt and present it in the same CM history.
        Only till_good can correct earlier turns. The returned CM trajectory is a snapshot.
        """
        if mode not in ("till_good", "best_of_n", "pass_at_k"):
            raise ValueError(f"Unknown iterative_correction mode: {mode}")
        if not iid_sampling and mode != "till_good":
            assert (
                self.correction_scope == "last_assistant"
            ), f"{mode} requires correction_scope='last_assistant'"
        assert rollout_num > 0 and self.max_correction_attempts > 0
        policy_chat = policy_session.chat_api
        messages = policy_session.trajectory["messages"]
        tools = policy_session.trajectory.get("tools", [])
        if messages[-1]["role"] == "assistant":
            first = _panda_dialog(messages, tools, {}, policy_chat)
        else:
            first = _generate_new_dialog(messages, tools, policy_chat)
        dialogs = {1: first}
        cm_traj = build_initial_cm_traj(
            first,
            self.base_cm_traj,
            extra_context,
            None if iid_sampling else self.correction_scope,
        )
        prompt_length = policy_prompt_length(first["messages"])
        prompt = first["messages"][:prompt_length]
        with mxlm.AgentSession(
            self.chat_correcting, cm_traj, self.tool_name_to_call
        ) as session:
            if not iid_sampling:
                adapter_policy = FindAndReplaceCorrectionAdapter(
                    tokenizer=policy_chat.name_or_path,
                    response_template=policy_chat.response_template,
                    max_replacement_tokens=self.max_replacement_tokens,
                )
            key = 1
            tool_content = None
            while True:
                if not iid_sampling:
                    step = self.correct_and_rollout(
                        session,
                        dialogs,
                        policy_chat,
                        adapter_policy,
                        tool_content,
                        allow_rollout=len(dialogs) < rollout_num,
                    )
                    key, tool_content = step["dialog_key"], step["tool_content"]
                    if tool_content is not None:
                        continue
                if mode == "till_good" or len(dialogs) == rollout_num:
                    if not iid_sampling:
                        outcome = (
                            "is_good=true"
                            if step["is_good"]
                            else "policy rollout budget exhausted"
                        )
                        session.trajectory["messages"].append(
                            dict(
                                role="tool",
                                name="correct_once",
                                tool_call_id=session.pending_tool_calls[0]["id"],
                                content=f"dialog_key={key}: {outcome}",
                            )
                        )
                        session.trajectory.pop("loop_finish_reason", None)
                    break

                new_key = max(dialogs) + 1
                dialogs[new_key] = _generate_new_dialog(prompt, tools, policy_chat)
                if not iid_sampling:
                    (new_root,) = render_dialog(
                        dialogs[new_key],
                        new_key,
                        [],
                        self.correction_scope,
                        last_turn_only=True,
                    )
                    tool_content = (
                        f"<correcting_prompt>dialog_key={key}: is_good=true，上一轮评审已归档。"
                        f"以下是从原 policy prompt 独立生成的新答复（dialog_key={new_key}）。"
                        "从这条新答复开始，用 correct_once 纠正不妥之处，满意后提交 is_good=true。</correcting_prompt>\n"
                        + new_root["text"]
                    )
                key = new_key

            # Keep the transcript, not MCP connections, credentials or tool executors.
            correcting_trajectory = {
                k: deepcopy(session.trajectory[k])
                for k in ("messages", "tools", "loop_finish_reason")
                if k in session.trajectory
            }

        result = dict(
            iterative_correction_mode=mode,
            policy_model_name=policy_chat.model,
            correcting_model_name=self.chat_correcting.model,
            rollout_num=rollout_num,
            correcting_trajectory=correcting_trajectory,
        )
        if mode != "till_good":
            candidates = [d["messages"] for d in dialogs.values()]
            assert all(
                ms[: policy_prompt_length(ms)] == prompt for ms in candidates
            ), f"{mode} candidates must share the original policy prompt"
        if mode == "best_of_n":
            context = render_dialog(dict(messages=deepcopy(prompt)), 1, [], None)
            if extra_context is not None:
                context.append(
                    dict(
                        type="text",
                        text=f"\n<correcting_prompt>\n{extra_context}\n</correcting_prompt>",
                    )
                )
            choice = BestOfNMixin.judge_candidates(
                self.chat_correcting,
                candidates,
                tools,
                [dict(role="user", content=context)],
                reply_start=prompt_length,
            )
            key = list(dialogs)[choice["best_idx"]]
            result.update(
                best_of_n_scores=dict(
                    zip(map(str, dialogs), choice["best_of_n_scores"])
                ),
                best_of_n_info=choice["best_of_n_info"],
            )
        if mode != "pass_at_k":
            result.update(dialog_key=key, corrected_messages=dialogs[key]["messages"])
        result["panda_json"] = dict(
            version="2.0",
            uuid=str(uuid4()),
            update_time=time.time_ns() // 1_000_000,
            dialogs={str(k): v for k, v in dialogs.items()},
            hash_map={},
            deleted_dialogs={},
        )
        return result


if __name__ == "__main__":
    import boxx
    from boxx import tree
    from onpanda.correcting_model.correcting_model import (
        build_correcting_model_with_policy,
    )
    from onpanda.test_utils import get_test_trajectories

    config = build_correcting_model_with_policy()
    trajectory = get_test_trajectories("bad_argument_arg2")
    policy_session = mxlm.AgentSession(config["chat_policy"], trajectory)
    agent = CorrectingAgent(
        config["correcting_model"].chat_correcting,
        max_replacement_tokens=config["adapter_policy"].max_replacement_tokens,
    )
    corrected = agent.iterative_correction(
        policy_session, rollout_num=3, mode="best_of_n"
    )
    panda_json = corrected["panda_json"]
    correcting_trajectory = corrected["correcting_trajectory"]
    tree(corrected)

    tmp_prefix = f"/tmp/{boxx.localTimeStr()}"

    boxx.savejson(corrected, tmp_prefix + ".corrected.json")
    boxx.savejson(
        corrected["correcting_trajectory"], tmp_prefix + ".correcting.panda.json"
    )

    boxx.savejson(corrected["panda_json"], tmp_prefix + ".policy.panda.json")
