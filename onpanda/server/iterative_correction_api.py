import argparse
import copy
import json
import queue
import threading
import time
import traceback
import urllib.parse

import requests
from flask import Flask, Response, jsonify, request, stream_with_context


ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
HEARTBEAT_INTERVAL_SECONDS = 600
UPSTREAM_TIMEOUT = (10, 4 * 60 * 60)


def decode_url_config_path(path_str, separator=",", assignor="@"):
    result = {}
    if not path_str:
        return result

    pairs = path_str.split(separator)
    for pair in pairs:
        if assignor not in pair:
            continue
        full_key, encoded_val = pair.split(assignor, 1)
        val_str = urllib.parse.unquote(encoded_val)
        try:
            if "." in val_str:
                value = float(val_str)
            else:
                value = int(val_str)
        except ValueError:
            value = val_str
            low = value.lower()
            if low == "true":
                value = True
            elif low == "false":
                value = False

        keys = full_key.split(".")
        current = result
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    return result


def deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def split_csv_arg(value):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_url_config(value):
    if not value:
        return {}
    s = value.strip()
    if not s:
        return {}
    if s.startswith("{"):
        return json.loads(s)
    return decode_url_config_path(s)


def build_upstream_headers(api_key):
    headers = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in {"host", "content-length", "connection"}:
            continue
        headers[k] = v
    if api_key and api_key != "no-key":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_upstream_url(base_url, subpath):
    base = base_url.rstrip("/")
    if subpath:
        return f"{base}/{subpath.lstrip('/')}"
    return base


def normalize_response_headers(headers):
    blocked = {"content-length", "transfer-encoding", "content-encoding", "connection"}
    out = {}
    for k, v in headers.items():
        if k.lower() in blocked:
            continue
        out[k] = v
    return out


def create_app(base_urls, api_keys, cli_config):
    app = Flask(__name__)

    default_base_url = base_urls[0]
    default_api_key = api_keys[0] if api_keys else ""
    correct_model_holder = {"model": None}

    @app.before_request
    def handle_cors_preflight():
        if request.method == "OPTIONS":
            return Response(status=204)

    @app.after_request
    def add_cors_headers(resp):
        req_headers = request.headers.get("Access-Control-Request-Headers", "*")
        req_method = request.headers.get("Access-Control-Request-Method", "*")
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = req_headers
        resp.headers["Access-Control-Allow-Methods"] = req_method
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp

    def proxy_plain(subpath, base_url, api_key):
        url = build_upstream_url(base_url, subpath)
        headers = build_upstream_headers(api_key)
        qs = request.query_string.decode("utf-8")
        if qs:
            url = f"{url}?{qs}"

        upstream = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            allow_redirects=False,
            timeout=UPSTREAM_TIMEOUT,
            stream=False,
        )
        resp_headers = normalize_response_headers(upstream.headers)
        return Response(
            upstream.content, status=upstream.status_code, headers=resp_headers
        )

    def proxy_chat_with_heartbeat(subpath, base_url, api_key, cli_config, url_config):
        body = json.loads(request.get_data(as_text=True) or "{}")
        if body.get("prompt_logprobs") and int(body.get("max_tokens", 0)) <= 1:
            return proxy_plain(subpath, base_url, api_key)

        correction_config = deep_merge(cli_config, url_config)
        print(
            "[iterative_correction] correction_config="
            + json.dumps(correction_config, ensure_ascii=False),
            flush=True,
        )

        result_q = queue.Queue()
        stream_flag = bool(body.get("stream"))
        req_model = body.get("model", "")
        req_messages = body.get("messages", [])
        correction_n = int(correction_config.get("n", 5))
        auth_header = request.headers.get("Authorization", "")
        bearer_token = (
            auth_header[len("Bearer ") :].strip()
            if auth_header.startswith("Bearer ")
            else ""
        )
        policy_api_key = bearer_token or "no-key"

        def get_correct_model():
            if correct_model_holder["model"] is None:
                from onpanda.correcting_model.correcting_sft_model import (
                    build_test_correcting_sft_model,
                )

                correct_model_holder["model"] = build_test_correcting_sft_model()
            return correct_model_holder["model"]

        def worker():
            try:
                import mxlm

                correct_model = get_correct_model()
                result_q.put(
                    (
                        "meta",
                        (
                            200,
                            {
                                "Content-Type": (
                                    "text/event-stream"
                                    if stream_flag
                                    else "application/json"
                                )
                            },
                        ),
                    )
                )

                policy_kwargs = dict(
                    base_url=base_url,
                    api_key=policy_api_key,
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
                created = int(time.time())
                response_id = f"chatcmpl-iterative-{created}"
                model_name = req_model or ""
                if stream_flag:
                    result_q.put(("chunk", b": iterative-correction-stream-start\n\n"))
                    chunk_obj = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": final_message.get("content", ""),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "iterative_correction": corrected,
                    }
                    result_q.put(
                        (
                            "chunk",
                            f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode(
                                "utf-8"
                            ),
                        )
                    )
                    result_q.put(("chunk", b"data: [DONE]\n\n"))
                else:
                    result_obj = {
                        "id": response_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": final_message.get("content", ""),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "iterative_correction": corrected,
                    }
                    result_q.put(
                        (
                            "chunk",
                            json.dumps(result_obj, ensure_ascii=False).encode("utf-8"),
                        )
                    )
                result_q.put(("done", None))
            except Exception as e:
                print(
                    "[iterative_correction] traceback:\n" + traceback.format_exc(),
                    flush=True,
                )
                payload = {"error": {"message": str(e), "type": "proxy_error"}}
                result_q.put(("error", payload))
                result_q.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

        first_kind = None
        first_payload = None
        while first_kind != "meta":
            first_kind, first_payload = result_q.get()
            if first_kind == "error":
                return jsonify(first_payload), 502
            if first_kind == "done":
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "iterative correction finished before response meta",
                                "type": "proxy_error",
                            }
                        }
                    ),
                    502,
                )

        upstream_status, upstream_headers = first_payload
        if stream_flag:
            upstream_headers["Cache-Control"] = "no-cache, no-transform"
            upstream_headers["X-Accel-Buffering"] = "no"
            upstream_headers["Connection"] = "keep-alive"

        @stream_with_context
        def generate():
            heartbeat = b" \n"
            if stream_flag:
                # Send an immediate SSE comment frame so devtools can see the stream opened.
                yield b": stream-open\n\n"
            while True:
                try:
                    kind, payload = result_q.get(timeout=HEARTBEAT_INTERVAL_SECONDS)
                    if kind == "chunk":
                        yield payload
                    elif kind == "error":
                        yield json.dumps(payload).encode("utf-8")
                    elif kind == "done":
                        break
                except queue.Empty:
                    yield heartbeat

        return Response(generate(), status=upstream_status, headers=upstream_headers)

    def handle_models_aggregate(subpath):
        if subpath != "models" or request.method not in {"GET", "POST"}:
            return None

        merged = []
        for idx, base_url in enumerate(base_urls):
            api_key = api_keys[idx] if idx < len(api_keys) else ""
            headers = build_upstream_headers(api_key)
            url = build_upstream_url(base_url, "models")
            try:
                r = requests.get(url, headers=headers, timeout=UPSTREAM_TIMEOUT)
                data = r.json()
                items = data.get("data", []) if isinstance(data, dict) else []
                merged.extend(items)
            except Exception:
                continue

        return jsonify({"object": "list", "data": merged})

    def route_common(subpath, cli_config, url_config):
        models_resp = handle_models_aggregate(subpath)
        if models_resp is not None:
            return models_resp

        if subpath == "chat/completions":
            return proxy_chat_with_heartbeat(
                subpath,
                default_base_url,
                default_api_key,
                cli_config,
                url_config,
            )

        return proxy_plain(subpath, default_base_url, default_api_key)

    @app.route(
        "/iterative_correction/v1", defaults={"subpath": ""}, methods=ALL_METHODS
    )
    @app.route(
        "/iterative_correction/v1/", defaults={"subpath": ""}, methods=ALL_METHODS
    )
    @app.route("/iterative_correction/v1/<path:subpath>", methods=ALL_METHODS)
    def proxy_iterative_empty(subpath):
        return route_common(subpath, cli_config, {})

    @app.route("/iterative_correction/<path:url_config_and_v1>", methods=ALL_METHODS)
    def proxy_iterative(url_config_and_v1):
        marker = "/v1/"
        if marker in url_config_and_v1:
            url_config_path, subpath = url_config_and_v1.split(marker, 1)
        elif url_config_and_v1.endswith("/v1"):
            url_config_path = url_config_and_v1[: -len("/v1")]
            subpath = ""
        else:
            return jsonify({"error": "path must include /v1/"}), 400

        url_config = decode_url_config_path(url_config_path)
        return route_common(subpath, cli_config, url_config)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        required=True,
        help="Comma-separated upstream base URLs. List mode is supported, e.g. http://a/v1,http://b/v1",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Comma-separated API keys. In list mode, length must match --base_url, e.g. k1,k2",
    )
    parser.add_argument(
        "--default_url_config",
        default="",
        help='Default URL config in JSON or url_config-path format, e.g. \'{"model":"CorrectingModle"}\' or model@CorrectingModle',
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

    base_urls = split_csv_arg(args.base_url)
    api_keys = split_csv_arg(args.api_key)
    if not api_keys:
        api_keys = [""] * len(base_urls)

    if len(base_urls) != len(api_keys):
        raise ValueError("base_url and api_key must have same length")

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
        "  - Merge in hijack code: correction_config = deep_merge(cli_config, url_config)"
    )
    print("  - URL config overrides CLI config")
    print("  - List mode: --base_url and --api_key support comma-separated lists")
    print("  - List mode rule: number of base URLs must equal number of API keys")
    print("[iterative_correction_api] examples:")
    print(
        f"  - curl http://{args.host}:{args.port}/iterative_correction/model@CorrectingModle/v1/models"
    )
    print(
        "  - python -m onpanda.server.iterative_correction_api "
        "--base_url http://127.0.0.1:9200/v1,http://127.0.0.1:9201/v1 "
        "--api_key key1,key2 --default_url_config model@CorrectingModle --model CorrectingModle"
    )
    app = create_app(base_urls=base_urls, api_keys=api_keys, cli_config=cli_config)
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=args.debug,
        use_reloader=args.debug,
    )


if __name__ == "__main__":
    main()
