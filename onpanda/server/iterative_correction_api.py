import argparse
import copy
import json

from flask import Flask
from mxlm import ChatAPI, decode_url_config_path, hijack_chat_api


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
    return decode_url_config_path(s)


def create_app(base_url, api_key, cli_config):
    app = Flask(__name__)
    default_base_url = base_url
    default_api_key = api_key
    correct_model_holder = {"model": None}

    def get_correct_model():
        if correct_model_holder["model"] is None:
            from onpanda.correcting_model.correcting_sft_model import (
                build_test_correcting_sft_model,
            )

            correct_model_holder["model"] = build_test_correcting_sft_model()
        return correct_model_holder["model"]

    def iterative_correction_process_func(body, headers, url_config):
        """
        Return dict:
        - direct_forward: bool
        - message: dict, contains role/content/tool_calls/reasoning*
        - extra_info: dict, merged into response root
        """
        if body.get("prompt_logprobs") and int(body.get("max_tokens", 0)) <= 1:
            return {"direct_forward": True}

        correction_config = deep_merge(cli_config, url_config)
        print(
            "[iterative_correction] correction_config="
            + json.dumps(correction_config, ensure_ascii=False),
            flush=True,
        )

        req_messages = body.get("messages", [])
        req_model = body.get("model", "")
        correction_n = int(correction_config.get("n", 5))

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

        chat_policy = ChatAPI(**policy_kwargs)
        correct_model = get_correct_model()
        corrected = correct_model.correcting_sampling(
            copy.deepcopy(req_messages),
            chat_policy,
            n=correction_n,
        )

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

    hijack_chat_api(
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


def main():
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
        help='Default URL config in JSON or url_config-path format, e.g. \'{"model":"CorrectingModle"}\' or model@CorrectingModle',
    )
    parser.add_argument("--model", default="", help="Default model, merged into cli_config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug and auto-reload")
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
    print("  - Merge in process_func: correction_config = deep_merge(cli_config, url_config)")
    print("  - URL config overrides CLI config")
    print("  - Aggregation moved to: python -m mxlm.aggregate_apis")
    print("[iterative_correction_api] examples:")
    print(
        f"  - curl http://{args.host}:{args.port}/iterative_correction/model@CorrectingModle/v1/models"
    )
    print(
        "  - python -m onpanda.server.iterative_correction_api "
        "--base_url http://127.0.0.1:9200/v1 "
        "--api_key key1 --default_url_config model@CorrectingModle --model CorrectingModle"
    )

    app = create_app(base_url=base_url, api_key=api_key, cli_config=cli_config)
    app.run(host=args.host, port=args.port, threaded=True, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
