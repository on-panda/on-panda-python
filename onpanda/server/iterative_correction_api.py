import argparse
import copy
import json

"""
Iterative correction proxy API.

`correcting_config` is merged from CLI `--default_url_config` and route `url_config`.
Supported keys include:
- `rollout_num` (int): iterative correction rollouts, default 5.
- `mode` (str): iterative correction mode, default "till_good".
- `chat` (dict): kwargs for `mxlm.ChatAPI` used by correcting model.
- `far` (dict): kwargs for `onpanda.FindAndReplaceCorrectionAdapter`.

url_config-path examples:
- rollout_num@3,chat.model@step1f-correct-sft-it1200
- rollout_num@3,chat.model@step1f-correct-sft-it1200,far.tokenizer@utf8_tokenizer
"""

from flask import Flask
import mxlm


UPSTREAM_TIMEOUT = (10, 4 * 60 * 60)
HEARTBEAT_INTERVAL_SECONDS = 600


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


def create_onpanda_app(base_url, api_key, cli_config):
    app = Flask(__name__)
    default_base_url = base_url
    default_api_key = api_key
    correcting_model_holder = {}

    def get_correcting_model(correcting_config, chat_policy):
        from onpanda.correcting_model.correcting_model import (
            build_test_correcting_model,
        )
        import onpanda

        model_config = {
            "far": correcting_config.get("far"),
            "chat": correcting_config.get("chat"),
        }
        model_key = json.dumps(model_config, sort_keys=True, ensure_ascii=False)
        correcting_model = correcting_model_holder.get(model_key)
        if correcting_model is None:
            far = model_config["far"]
            chat = model_config["chat"]
            chat_correcting = None
            if far is not None:
                far = onpanda.FindAndReplaceCorrectionAdapter(**far)
            if chat is not None:

                def reasoning_parser(message):
                    content = message["content"]
                    splitter = correcting_config.get(
                        "reasoning_end_splitter", "\n</think>\n"
                    )
                    idx = content.rfind(splitter)
                    if idx != -1:
                        message["content"] = content[idx + len(splitter) :]
                        message["reasoning"] = content[:idx]
                    return message

                chat.setdefault("parser", reasoning_parser)
                # chat_correcting using same base_url and api_key as chat_policy
                chat.setdefault("base_url", chat_policy.base_url)
                chat.setdefault("api_key", chat_policy.api_key)
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
        - rollout_num: correction rollouts, default 5
        - mode: iterative mode, default till_good
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
        req_model = body.get("model", "")
        rollout_num = int(correcting_config.get("rollout_num", 5))
        mode = correcting_config.get("mode", "till_good")

        auth_header = headers.get("Authorization", "")
        bearer_token = (
            auth_header[len("Bearer ") :].strip()
            if auth_header.startswith("Bearer ")
            else ""
        )
        selected_base_url = default_base_url

        policy_kwargs = dict(
            base_url=selected_base_url,
            api_key=bearer_token or "no-key",
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
        # print("chat_policy created:", chat_policy, correcting_model.chat_correcting, flush=True)
        corrected = correcting_model.iterative_correction(
            copy.deepcopy(req_messages),
            chat_policy,
            rollout_num=rollout_num,
            mode=mode,
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
        api_key=default_api_key,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        upstream_timeout=UPSTREAM_TIMEOUT,
        enable_cors=True,
    )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        required=True,
        help="Single upstream base URL, e.g. http://127.0.0.1:9200/v1",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Optional single API key",
    )
    parser.add_argument(
        "--default_url_config",
        default="",
        help=(
            "Default URL config in url_config-path format. "
            "Example: rollout_num@3,chat.model@step1f-correct-sft-it1200,"
            "far.tokenizer@utf8_tokenizer"
        ),
    )
    parser.add_argument(
        "--model", default="", help="Default model, merged into cli_config"
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

    print(f"[iterative_correction_api] listening on http://{args.host}:{args.port}")
    print("[iterative_correction_api] routes:")
    print("  - /iterative_correction/v1/* (empty url_config)")
    print("  - /iterative_correction/{url_config}/v1/*")
    print("[iterative_correction_api] config:")
    print("  - CLI config: from --default_url_config and --model")
    print("  - URL config: url_config in /iterative_correction/{url_config}/v1/*")
    print(
        "  - Merge in process_func: correcting_config = deep_merge(cli_config, url_config)"
    )
    print("  - Common correcting_config keys: rollout_num, mode, chat.*, far.*")
    print(
        "  - chat.* -> mxlm.ChatAPI(**chat), far.* -> onpanda.FindAndReplaceCorrectionAdapter(**far)"
    )
    print("  - URL config overrides CLI config")
    print("  - Aggregation moved to: python -m mxlm.aggregate_apis")
    print("[iterative_correction_api] examples:")
    print(
        f"  - curl http://{args.host}:{args.port}/iterative_correction/chat.model@model_name/v1/models"
    )
    print(
        f"  - curl http://{args.host}:{args.port}/iterative_correction/rollout_num@3,chat.model@step1f-correct-sft-it1200,far.tokenizer@utf8_tokenizer/v1/models"
    )
    print(
        "  - python -m onpanda.server.iterative_correction_api "
        "--base_url http://127.0.0.1:9200/v1 "
        "--api_key key1 "
        "--default_url_config rollout_num@3,chat.model@step1f-correct-sft-it1200,far.tokenizer@utf8_tokenizer "
        "--model model_name"
    )

    app = create_onpanda_app(base_url=base_url, api_key=api_key, cli_config=cli_config)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=args.debug,
        use_reloader=args.debug,
    )
