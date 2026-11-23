"""
Iterative correction proxy API.

`correcting_config` is merged from CLI `--default_url_config` and route `url_config`.
Supported keys include:
- `model` (str): default policy model if request body omits `model`.
- `rollout_num` (int): iterative correction rollouts, default 5.
- `mode` (str): iterative correction mode, default "till_good".
- `iid_sampling` (bool): sample each rollout independently, default False.
- `eval_name` (str): unique name for one pass_at_k evaluation run; each prompt hash gets one batch by default.
- `allow_repeat` (bool): for pass_at_k only, resample after a prompt consumes k cached candidates.
- `rerun_bypass` (bool): direct-forward single-turn rerun extraction requests, default False.
- `chat` (dict): kwargs for `mxlm.ChatAPI` used by correcting model.
- `far` (dict): correcting adapter kwargs; its response template defaults to `onpanda/default`.
- `name_or_path` (str): set the policy response template explicitly to continue reasoning or tool calls.
- `policy_far` (dict): policy adapter settings; `tokenizer` and `max_replacement_tokens`
  control replacement alignment. The policy ChatAPI owns its response template.

url_config-path examples:
- SFT correcting model for best_of_n mode:
    - rollout_num@4,mode@best_of_n,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false
- reasoning model as correcting model:
    - rollout_num@3,chat.model@step-3.7-flash,far.max_replacement_tokens@1,chat.is_reasoning@true
- correcting model as reward model:
    - rollout_num@3,mode@best_of_n,iid_sampling@true,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false
- agent policy (reasoning and tool calls), whose response template name needs url encoding:
    - rollout_num@3,chat.model@step-3.7-flash,chat.is_reasoning@true,name_or_path@Qwen%2FQwen3.6-35B-A3B
"""

import argparse
import copy
import json
import os
import textwrap
import threading
from urllib.parse import quote

import mxlm

UPSTREAM_TIMEOUT = (10, 4 * 60 * 60)
HEARTBEAT_INTERVAL_SECONDS = 600
PASS_AT_K_TASKS = {}
PASS_AT_K_TASKS_LOCK = threading.Lock()

DUMP_FILE_LOCKS = {}
DUMP_FILE_LOCKS_GUARD = threading.Lock()


def dump_corrected_log(corrected, *, log_dir, prompt_hash, eval_name=None):
    if not log_dir:
        return
    if eval_name:
        root = os.path.realpath(log_dir)
        log_dir = os.path.realpath(os.path.join(root, str(eval_name)))
        if os.path.commonpath([root, log_dir]) != root:
            raise ValueError("eval_name cannot escape the configured log directory")
    os.makedirs(log_dir, exist_ok=True)
    target_path = os.path.join(log_dir, f"{quote(prompt_hash, safe='%')}.json")
    with DUMP_FILE_LOCKS_GUARD:
        lock = DUMP_FILE_LOCKS.setdefault(target_path, threading.Lock())
    with lock:
        entries = []
        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        entries.append(corrected)
        tmp_path = f"{target_path}.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target_path)


def deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def parse_url_config(value):
    if not value:
        return {}
    s = value.strip()
    if not s:
        return {}
    if s.startswith("{"):
        return json.loads(s)
    return mxlm.decode_url_config_path(s)


def create_onpanda_app(base_url, api_key, cli_config, disable_auth=False, log_dir=None):
    correcting_model_holder = {}

    def get_correcting_model(correcting_config, chat_policy):
        import onpanda
        from onpanda.correcting_model.correcting_model import (
            build_test_correcting_model,
        )

        far = copy.deepcopy(correcting_config.get("far"))
        policy_far = copy.deepcopy(correcting_config.get("policy_far"))
        if policy_far is not None:
            policy_far["response_template"] = chat_policy.response_template
        chat = copy.deepcopy(correcting_config.get("chat"))
        if chat is not None:
            # chat_correcting uses the same upstream by default, while still
            # allowing chat.base_url/chat.api_key overrides in correcting_config.
            chat.setdefault("base_url", chat_policy.base_url)
            chat.setdefault("api_key", chat_policy.api_key)
        model_config = {
            "far": far,
            "policy_far": policy_far,
            "chat": chat,
        }
        model_key = json.dumps(
            model_config, sort_keys=True, ensure_ascii=False, default=repr
        )
        correcting_model, adapter_policy = correcting_model_holder.get(
            model_key, (None, None)
        )
        if correcting_model is None:
            chat_correcting = None
            if far is not None:
                far = onpanda.FindAndReplaceCorrectionAdapter(**far)
            if policy_far is not None:
                adapter_policy = onpanda.FindAndReplaceCorrectionAdapter(**policy_far)
            if chat is not None:

                def reasoning_parser(message):
                    content = message.get("content")
                    if not isinstance(content, str):
                        return message
                    splitter = correcting_config.get(
                        "reasoning_end_splitter", "</think>"
                    )
                    idx = content.rfind(splitter)
                    if idx != -1:
                        message["reasoning"] = content[:idx].rstrip()
                        message["content"] = content[idx + len(splitter) :].lstrip()
                    return message

                chat.setdefault("parser", reasoning_parser)
                chat_correcting = mxlm.ChatAPI(**chat)
            correcting_model = build_test_correcting_model(
                chat_correcting=chat_correcting,
                adapter=far,
            )
            correcting_model_holder[model_key] = (correcting_model, adapter_policy)
        return correcting_model, adapter_policy

    def run_correction(config, chat_policy, messages, tools, **options):
        model, adapter_policy = get_correcting_model(config, chat_policy)
        corrected = model.iterative_correction(
            copy.deepcopy(messages),
            chat_policy,
            tools=tools,
            adapter_policy=adapter_policy,
            **options,
        )
        if options["mode"] == "pass_at_k":
            candidates = [
                step["corrected_messages"][-1]
                for chain in corrected["correction_till_goods"]
                for step in chain["correction_steps"]
            ]
        else:
            final = corrected.get("corrected_messages") or [
                {"role": "assistant", "content": ""}
            ]
            candidates = [final[-1]]
        return corrected, candidates

    return _create_correction_app(
        base_url,
        api_key,
        cli_config,
        disable_auth,
        log_dir,
        hijack_path="iterative_correction",
        run_correction=run_correction,
    )


def _create_correction_app(
    base_url,
    api_key,
    cli_config,
    disable_auth,
    log_dir,
    *,
    hijack_path,
    run_correction,
    validate_request=None,
):
    """Share the HTTP protocol; each backend returns a JSON log and messages to issue."""
    from flask import Flask, request

    app = Flask(__name__)
    if disable_auth:
        # Direct-forwarded requests must ignore the caller's bearer even without a server key.
        @app.before_request
        def ignore_request_bearer():
            request.environ.pop("HTTP_AUTHORIZATION", None)

    def build_hijack_result(final_message, iterative_correction_info):
        finish_reason = final_message.get("finish_reason")
        if (
            finish_reason == "stop"
            and final_message.get("tool_calls")
            and final_message["tool_calls"] != [{}]
        ):
            finish_reason = "tool_calls"
        elif not finish_reason:
            finish_reason = (
                "tool_calls"
                if final_message.get("tool_calls")
                and final_message["tool_calls"] != [{}]
                else "stop"
            )
        message = {
            "role": final_message.get("role", "assistant"),
            "content": final_message.get("content", ""),
        }
        for key in ("tool_calls", "reasoning", "reasoning_content"):
            if key in final_message:
                message[key] = final_message[key]
        return {
            "direct_forward": False,
            "message": message,
            "finish_reason": finish_reason,
            "extra_info": {"iterative_correction": iterative_correction_info},
        }

    def iterative_correction_process_func(body, headers, url_config):
        """Intercept chat/completions; other paths are direct-forwarded by mxlm."""
        if validate_request is not None:
            validate_request(body, url_config)
        if body.get("prompt_logprobs") and int(body.get("max_tokens", 10000)) <= 1:
            return {"direct_forward": True}

        correcting_config = deep_merge(cli_config, url_config)

        req_messages = body.get("messages", [])
        rerun_bypass = bool(correcting_config.get("rerun_bypass", False))
        # Shortcut single-turn rerun extraction before correction state is touched.
        if (
            rerun_bypass
            and sum(message["role"] == "user" for message in req_messages) > 1
        ):
            return {"direct_forward": True}

        req_model = body.get("model") or correcting_config.get("model", "")
        rollout_num = int(correcting_config.get("rollout_num", 5))
        mode = correcting_config.get("mode", "till_good")
        iid_sampling = correcting_config.get("iid_sampling", False)
        allow_repeat = bool(correcting_config.get("allow_repeat", False))
        eval_name = correcting_config.get("eval_name")
        if mode == "pass_at_k":
            if not eval_name:
                raise ValueError("pass_at_k mode requires eval_name in url config")
            eval_name = str(eval_name)
        # TODO: support structured response_format in iterative correction.
        tools = body.get("tools")
        prompt_object = req_messages
        while prompt_object and prompt_object[-1]["role"] == "assistant":
            prompt_object = prompt_object[:-1]
        if tools:
            # Same messages with different tools are different prompts for pass_at_k and logs.
            prompt_object = [prompt_object, tools]
        prompt_hash = mxlm.hash_object_sha256_base64(prompt_object)

        auth_header = headers.get("Authorization", "")
        bearer_token = (
            auth_header[len("Bearer ") :].strip()
            if auth_header.startswith("Bearer ")
            else ""
        )
        policy_kwargs = {
            "base_url": base_url,
            "model": req_model,
            "name_or_path": correcting_config.get("name_or_path"),
        }
        for k in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "chat_template_kwargs",  # e.g. enable_thinking of a reasoning policy
        ):
            if k in body:
                policy_kwargs[k] = body[k]

        def generate():
            corrected, candidates = run_correction(
                correcting_config,
                mxlm.ChatAPI(
                    **policy_kwargs,
                    api_key=(api_key if disable_auth else bearer_token) or "no-key",
                ),
                req_messages,
                tools,
                rollout_num=rollout_num,
                mode=mode,
                iid_sampling=iid_sampling,
            )
            expected = rollout_num if mode == "pass_at_k" else 1
            if len(candidates) != expected:
                raise ValueError(
                    f"Expected {expected} correction candidates, got {len(candidates)}"
                )
            dump_corrected_log(
                corrected,
                log_dir=log_dir,
                prompt_hash=prompt_hash,
                eval_name=eval_name,
            )
            return corrected, candidates

        if mode == "pass_at_k":
            with PASS_AT_K_TASKS_LOCK:
                task_state = PASS_AT_K_TASKS.setdefault(
                    eval_name,
                    {"k": rollout_num, "prompt_states": {}},
                )
                if int(task_state["k"]) != rollout_num:
                    raise ValueError(
                        "pass_at_k k mismatch for eval_name "
                        f"{eval_name}: existing={task_state['k']}, current={rollout_num}"
                    )
                prompt_state = task_state["prompt_states"].setdefault(
                    prompt_hash,
                    {"lock": threading.Lock(), "times": 0, "candidates": []},
                )

            with prompt_state["lock"]:
                if not allow_repeat and int(prompt_state["times"]) >= rollout_num:
                    raise ValueError(
                        "pass_at_k prompt request times exceeded k: "
                        f"eval_name={eval_name}, prompt_hash={prompt_hash}, "
                        f"times={prompt_state['times']}, k={rollout_num}"
                    )
                if prompt_state["candidates"]:
                    extra_info = (
                        f"{rollout_num - len(prompt_state['candidates']) + 1}/"
                        f"{rollout_num} of "
                        f"{hijack_path} cached pass@k candidates"
                    )
                else:
                    extra_info, prompt_state["candidates"] = generate()

                selected_message = prompt_state["candidates"].pop(0)
                prompt_state["times"] += 1

            return build_hijack_result(selected_message, extra_info)

        corrected, candidates = generate()
        return build_hijack_result(candidates[0], corrected)

    mxlm.hijack_chat_api(
        app,
        hijack_path=hijack_path,
        process_func=iterative_correction_process_func,
        base_url=base_url,
        api_key=api_key if disable_auth else None,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        upstream_timeout=UPSTREAM_TIMEOUT,
        enable_cors=True,
    )

    return app


def _serve(create_app, hijack_path, description, example_config):
    parser = argparse.ArgumentParser(
        description=description.strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_url",
        required=True,
        help="Single upstream base URL, e.g. http://127.0.0.1:9200/v1",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Optional upstream API key used when --disable_auth is set",
    )
    parser.add_argument(
        "--disable_auth",
        action="store_true",
        help="Ignore request Bearer tokens when calling the upstream API",
    )
    parser.add_argument(
        "--default_url_config",
        default="",
        help=f"Default correcting config as JSON or url_config-path. Example: {example_config}",
    )
    parser.add_argument(
        "--model", default="", help="Default policy model if request body omits model"
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="Root directory for eval logs. If omitted, eval logs are not written.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug and auto-reload"
    )
    args = parser.parse_args()

    base_url = args.base_url.strip()
    api_key = args.api_key.strip() if args.api_key else None
    log_dir = args.log_dir.strip() or None

    cli_config = parse_url_config(args.default_url_config)
    if args.model:
        cli_config["model"] = args.model
    request_bearer_status = (
        "ignored" if args.disable_auth else "forwarded to upstream when present"
    )

    startup_message = textwrap.dedent(f"""
        [{hijack_path}_api] listening on http://{args.host}:{args.port}
        Routes: /{hijack_path}/v1/* and /{hijack_path}/{{url_config}}/v1/*
        Route options override --default_url_config and --model; see --help for supported keys.
        Eval logs: {log_dir or "disabled"}; filenames use the Panda prompt hash.
        Request Bearer: {request_bearer_status}
        Examples:
          curl http://{args.host}:{args.port}/{hijack_path}/{example_config}/v1/models
          python -m onpanda.server.{hijack_path}_api --base_url http://127.0.0.1:9200/v1 --default_url_config {example_config} --model policy-model
        """).strip()
    print(startup_message)

    app = create_app(
        base_url=base_url,
        api_key=api_key,
        cli_config=cli_config,
        disable_auth=args.disable_auth,
        log_dir=log_dir,
    )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=args.debug,
        use_reloader=args.debug,
    )


if __name__ == "__main__":
    _serve(
        create_onpanda_app,
        "iterative_correction",
        __doc__,
        "rollout_num@3,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false",
    )
