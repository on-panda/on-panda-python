r"""OpenAI-compatible chat completions backed by CorrectingAgent.

    python -m onpanda.server.correcting_agent_api --base_url http://localhost:9200/v1 \
        --default_url_config chat.model@tool-calling-cm,max_replacement_tokens@1

POST to /correcting_agent/v1/chat/completions, or override options on the route:
    /correcting_agent/mode@pass_at_k,eval_name@demo,rollout_num@3/v1/chat/completions

Options: model, name_or_path (policy tokenizer/template), rollout_num (5), mode
(till_good, best_of_n, pass_at_k), iid_sampling, eval_name, allow_repeat,
rerun_bypass, max_replacement_tokens (1), max_correction_attempts (5),
correction_scope (last_assistant only; ignored for IID), and extra_context (a CM hint,
not a policy kwarg).
Set name_or_path explicitly to continue policy reasoning or tool calls.
chat.model and chat sampling options configure the CM, which defaults to the policy
model/upstream. Only server configuration may set CM connection details, base_cm_traj
(including tool_configs), or tool executors. Policy tools are schemas, never executed.

As on the legacy route, iterative_correction in the response holds the run log
(or a cached-candidate notice). All modes use a single correcting_trajectory
(messages, CM tool schemas and optional loop_finish_reason, not live sessions or tool_configs).
"""

import mxlm

from onpanda.correcting_model.correction_call.trajectory import policy_prompt_length
from onpanda.server.iterative_correction_api import _create_correction_app, _serve

_URL_OPTIONS = {
    "model",
    "name_or_path",
    "rollout_num",
    "mode",
    "iid_sampling",
    "eval_name",
    "allow_repeat",
    "rerun_bypass",
    "max_replacement_tokens",
    "max_correction_attempts",
    "correction_scope",
    "extra_context",
    "chat",
}
_CHAT_OPTIONS = {
    "model",
    "name_or_path",
    "is_reasoning",
    "reasoning_key",
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
}


def _validate_request(body, url_config):
    # Untrusted routes cannot redirect credentials, install tools or configure a local client.
    chat = url_config.get("chat", {})
    if (
        url_config.keys() - _URL_OPTIONS
        or not isinstance(chat, dict)
        or chat.keys() - _CHAT_OPTIONS
    ):
        raise ValueError(
            "Unsupported correcting_agent URL option; configure CM connections/tools on the server"
        )
    if {"base_cm_traj", "tool_configs", "tool_name_to_call"}.intersection(body):
        raise ValueError("Only server configuration can supply tool executors")
    tools = body.get("tools") or []
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict) or tool.get("type") != "function" for tool in tools
    ):
        raise ValueError("Policy tools must be function schemas")


def _prompt_message(message):
    """Ignore only the CM's computed scope marker when comparing policy histories."""
    message = dict(message)
    sidecar = dict(message.pop("sidecar", {}))
    correction = dict(sidecar.pop("correction", {}))
    correction.pop("_correction_area", None)
    if correction:
        sidecar["correction"] = correction
    if sidecar:
        message["sidecar"] = sidecar
    return message


def _completion_for(prompt, candidate):
    if (
        candidate[-1].get("role") != "assistant"
        or [_prompt_message(m) for m in candidate[: policy_prompt_length(candidate)]]
        != prompt
    ):
        raise ValueError(
            "Cannot return a historical-turn fork as the completion of this prompt"
        )
    return candidate[-1]


def _run_correction(config, chat_policy, messages, tools, **options):
    from onpanda.correcting_model.correcting_agent import CorrectingAgent

    correction_scope = config.get("correction_scope", "last_assistant")
    if not options.get("iid_sampling") and correction_scope != "last_assistant":
        raise ValueError("Chat completions require correction_scope='last_assistant'")

    chat = dict(config.get("chat", {}))
    chat.setdefault("model", chat_policy.model)
    chat.setdefault("base_url", chat_policy.base_url)
    # A separate CM upstream must have its own configured key, not the caller's bearer.
    same_upstream = chat["base_url"].rstrip("/") == chat_policy.base_url.rstrip("/")
    chat.setdefault("api_key", chat_policy.api_key if same_upstream else "no-key")
    agent = CorrectingAgent(
        cm_chat=mxlm.ChatAPI(**chat),
        max_replacement_tokens=int(config.get("max_replacement_tokens", 1)),
        correction_scope=correction_scope,
        max_correction_attempts=int(config.get("max_correction_attempts", 5)),
        base_cm_traj=config.get("base_cm_traj"),
        tool_name_to_call=config.get("tool_name_to_call"),
    )
    policy_traj = {"messages": messages}
    if tools is not None:
        policy_traj["tools"] = tools
    with mxlm.AgentSession(chat_policy, policy_traj) as policy:
        corrected = agent.iterative_correction(
            policy, extra_context=config.get("extra_context"), **options
        )

    prompt = [_prompt_message(m) for m in messages[: policy_prompt_length(messages)]]
    if options["mode"] == "pass_at_k":
        histories = (d["messages"] for d in corrected["panda_json"]["dialogs"].values())
    else:
        histories = [corrected["corrected_messages"]]
    return corrected, [_completion_for(prompt, candidate) for candidate in histories]


def create_onpanda_app(base_url, api_key, cli_config, disable_auth=False, log_dir=None):
    return _create_correction_app(
        base_url,
        api_key,
        cli_config,
        disable_auth,
        log_dir,
        hijack_path="correcting_agent",
        run_correction=_run_correction,
        validate_request=_validate_request,
    )


if __name__ == "__main__":
    _serve(
        create_onpanda_app,
        "correcting_agent",
        __doc__,
        "rollout_num@3,chat.model@tool-calling-cm,max_replacement_tokens@1",
    )
