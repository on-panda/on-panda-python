"""
Iterative correction proxy API.

`correcting_config` is merged from CLI `--default_url_config` and route `url_config`.
Supported keys include:
- `model` (str): default policy model if request body omits `model`.
- `rollout_num` (int): iterative correction rollouts, default 5.
- `mode` (str): iterative correction mode, default "till_good".
- `iid_sampling` (bool): sample each rollout independently, default False.
- `eval_name` (str): in-memory cache namespace for pass_at_k mode.
- `chat` (dict): kwargs for `mxlm.ChatAPI` used by correcting model.
- `far` (dict): kwargs for `onpanda.FindAndReplaceCorrectionAdapter`.

url_config-path examples:
- SFT correcting model for best_of_n mode:
    - rollout_num@4,mode@best_of_n,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false
- reasoning model as correcting model:
    - rollout_num@3,chat.model@step-3.7-flash,far.max_replacement_tokens@1,chat.is_reasoning@true
- correcting model as reward model:
    - rollout_num@3,mode@best_of_n,iid_sampling@true,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false
"""

import argparse
import copy
import hashlib
import json
import threading
import textwrap

from flask import Flask
import mxlm

UPSTREAM_TIMEOUT = (10, 4 * 60 * 60)
HEARTBEAT_INTERVAL_SECONDS = 600
PASS_AT_K_TASKS = {}
PASS_AT_K_TASKS_LOCK = threading.Lock()


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


def create_onpanda_app(base_url, api_key, cli_config, disable_auth=False):
    app = Flask(__name__)
    default_base_url = base_url
    default_api_key = api_key
    correcting_model_holder = {}

    def get_correcting_model(correcting_config, chat_policy):
        from onpanda.correcting_model.correcting_model import (
            build_test_correcting_model,
        )
        import onpanda

        far = copy.deepcopy(correcting_config.get("far"))
        chat = copy.deepcopy(correcting_config.get("chat"))
        if chat is not None:
            # chat_correcting uses the same upstream by default, while still
            # allowing chat.base_url/chat.api_key overrides in correcting_config.
            chat.setdefault("base_url", chat_policy.base_url)
            chat.setdefault("api_key", chat_policy.api_key)
        model_config = {
            "far": far,
            "chat": chat,
        }
        model_key = json.dumps(
            model_config, sort_keys=True, ensure_ascii=False, default=repr
        )
        correcting_model = correcting_model_holder.get(model_key)
        if correcting_model is None:
            chat_correcting = None
            if far is not None:
                far = onpanda.FindAndReplaceCorrectionAdapter(**far)
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
            correcting_model_holder[model_key] = correcting_model
        return correcting_model

    def iterative_correction_process_func(body, headers, url_config):
        """
        `correcting_config` keys used here:
        - model: default policy model if the request body omits model
        - rollout_num: correction rollouts, default 5
        - mode: iterative mode, default till_good
        - iid_sampling: sample each rollout independently, default False
        - chat: used to build correcting chat, mxlm.ChatAPI(**chat)
        - far: used to build correcting adapter, FindAndReplaceCorrectionAdapter(**far)

        Return dict:
        - direct_forward: bool
        - message: dict, contains role/content/tool_calls/reasoning*
        - extra_info: dict, merged into response root
        """
        if body.get("prompt_logprobs") and int(body.get("max_tokens", 10000)) <= 1:
            return {"direct_forward": True}

        correcting_config = deep_merge(cli_config, url_config)
        print(
            "[iterative_correction] correcting_config="
            + json.dumps(correcting_config, ensure_ascii=False),
            flush=True,
        )

        req_messages = body.get("messages", [])
        req_model = body.get("model") or correcting_config.get("model", "")
        rollout_num = int(correcting_config.get("rollout_num", 5))
        mode = correcting_config.get("mode", "till_good")
        iid_sampling = correcting_config.get("iid_sampling", False)
        eval_name = correcting_config.get("eval_name")
        if mode == "pass_at_k":
            if not eval_name:
                raise ValueError("pass_at_k mode requires eval_name in url config")
            eval_name = str(eval_name)
        # TODO: support tool calls and structured response_format in iterative correction.
        assert "tools" not in body, "iterative_correction does not support tools yet"

        auth_header = headers.get("Authorization", "")
        bearer_token = (
            auth_header[len("Bearer ") :].strip()
            if auth_header.startswith("Bearer ")
            else ""
        )
        selected_base_url = default_base_url

        policy_kwargs = dict(
            base_url=selected_base_url,
            api_key=(default_api_key if disable_auth else bearer_token) or "no-key",
            model=req_model,
        )
        for k in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            if k in body:
                policy_kwargs[k] = body[k]

        chat_policy = mxlm.ChatAPI(**policy_kwargs)
        correcting_model = get_correcting_model(correcting_config, chat_policy)

        if mode == "pass_at_k":
            prompt_hash = hashlib.sha256(
                json.dumps(
                    {"messages": req_messages, "tools": body.get("tools", [])},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
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
                if int(prompt_state["times"]) >= rollout_num:
                    raise ValueError(
                        "pass_at_k prompt request times exceeded k: "
                        f"eval_name={eval_name}, prompt_hash={prompt_hash}, "
                        f"times={prompt_state['times']}, k={rollout_num}"
                    )
                if prompt_state["candidates"]:
                    extra_info = (
                        f"{prompt_state['times'] + 1}/{rollout_num} of "
                        "iterative_correction cached pass@k candidates"
                    )
                else:
                    extra_info = corrected = correcting_model.iterative_correction(
                        copy.deepcopy(req_messages),
                        chat_policy,
                        rollout_num=rollout_num,
                        mode=mode,
                        iid_sampling=iid_sampling,
                    )
                    try:
                        __import__("boxx").tree([eval_name, prompt_hash, corrected])
                    except ModuleNotFoundError:
                        pass
                    prompt_state["candidates"] = [
                        correction_step["corrected_messages"][-1]
                        for correction_till_good in corrected["correction_till_goods"]
                        for correction_step in correction_till_good["correction_steps"]
                    ]
                    assert len(prompt_state["candidates"]) == rollout_num

                selected_message = prompt_state["candidates"].pop(0)
                prompt_state["times"] += 1

            return {
                "direct_forward": False,
                "message": selected_message,
                "extra_info": {"iterative_correction": extra_info},
            }

        # print("chat_policy created:", chat_policy, correcting_model.chat_correcting, flush=True)
        corrected = correcting_model.iterative_correction(
            copy.deepcopy(req_messages),
            chat_policy,
            rollout_num=rollout_num,
            mode=mode,
            iid_sampling=iid_sampling,
        )
        try:
            __import__("boxx").tree(corrected)
        except ModuleNotFoundError:
            pass

        corrected_messages = corrected.get("corrected_messages", [])
        final_message = (
            corrected_messages[-1]
            if corrected_messages
            else {"role": "assistant", "content": ""}
        )
        message = {
            "role": final_message.get("role", "assistant"),
            "content": final_message.get("content", ""),
        }
        if "tool_calls" in final_message:
            message["tool_calls"] = final_message["tool_calls"]
        if "reasoning" in final_message:
            message["reasoning"] = final_message["reasoning"]
        if "reasoning_content" in final_message:
            message["reasoning_content"] = final_message["reasoning_content"]

        return {
            "direct_forward": False,
            "message": message,
            "extra_info": {"iterative_correction": corrected},
        }

    mxlm.hijack_chat_api(
        app,
        hijack_path="iterative_correction",
        process_func=iterative_correction_process_func,
        base_url=default_base_url,
        api_key=default_api_key if disable_auth else None,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        upstream_timeout=UPSTREAM_TIMEOUT,
        enable_cors=True,
    )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.strip(),
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
        help=(
            "Default correcting config in url_config-path format. "
            "Example: rollout_num@3,chat.model@peqwen3-sft-cm-it1000,"
            "chat.is_reasoning@false"
        ),
    )
    parser.add_argument(
        "--model", default="", help="Default policy model if request body omits model"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument(
        "--debug", action="store_true", help="Enable Flask debug and auto-reload"
    )
    args = parser.parse_args()

    base_url = args.base_url.strip()
    api_key = args.api_key.strip() if args.api_key else None

    cli_config = parse_url_config(args.default_url_config)
    if args.model:
        cli_config["model"] = args.model
    request_bearer_status = (
        "ignored" if args.disable_auth else "forwarded to upstream when present"
    )

    startup_message = textwrap.dedent(f"""
        [iterative_correction_api] listening on http://{args.host}:{args.port}
        [iterative_correction_api] routes:
          - /iterative_correction/v1/* (empty url_config)
          - /iterative_correction/{{url_config}}/v1/*
        [iterative_correction_api] config:
          - CLI config: from --default_url_config and --model
          - URL config: url_config in /iterative_correction/{{url_config}}/v1/*
          - Merge in process_func: correcting_config = deep_merge(cli_config, url_config)
          - Common correcting_config keys: rollout_num, mode, iid_sampling, chat.*, far.*
          - Request Bearer: {request_bearer_status}
          - chat.* -> mxlm.ChatAPI(**chat), far.* -> onpanda.FindAndReplaceCorrectionAdapter(**far)
          - URL config overrides CLI config
          - Aggregation moved to: python -m mxlm.aggregate_apis
          - If using reasoning model as correcting model i.e. `chat.is_reasoning@true` and non-utf8 tokenizer, set `far.max_replacement_tokens` to a small number like 1, otherwise the replacement token may inject too much information.
        [iterative_correction_api] examples:
          - curl http://{args.host}:{args.port}/iterative_correction/chat.model@model_name,chat.is_reasoning@false/v1/models
          - curl http://{args.host}:{args.port}/iterative_correction/rollout_num@3,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false,far.tokenizer@utf8_tokenizer/v1/models
          - python -m onpanda.server.iterative_correction_api --base_url http://127.0.0.1:9200/v1 --api_key key1 --default_url_config rollout_num@3,chat.model@peqwen3-sft-cm-it1000,chat.is_reasoning@false,far.tokenizer@utf8_tokenizer --model model_name
        """).strip()
    print(startup_message)

    app = create_onpanda_app(
        base_url=base_url,
        api_key=api_key,
        cli_config=cli_config,
        disable_auth=args.disable_auth,
    )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=args.debug,
        use_reloader=args.debug,
    )
