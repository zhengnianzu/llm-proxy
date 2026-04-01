import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

import anthropic
import httpx
from dotenv import load_dotenv
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs_anthropic"


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pick_request_file(request_file: Optional[str]) -> Path:
    if request_file:
        path = Path(request_file)
        if not path.is_absolute():
            path = ROOT_DIR / request_file
        if not path.exists():
            raise FileNotFoundError(f"request file not found: {path}")
        return path

    candidates = sorted(glob.glob(str(LOG_DIR / "*-req.json")))
    if not candidates:
        raise FileNotFoundError(f"no request files found under {LOG_DIR}")
    return Path(candidates[-1])


def load_request_payload(request_path: Path) -> dict:
    with request_path.open("r", encoding="utf-8") as f:
        body = json.load(f)

    allowed_keys = {
        "model",
        "max_tokens",
        "messages",
        "system",
        "tools",
        "temperature",
        "top_k",
        "top_p",
        "metadata",
        "stop_sequences",
        "stream",
        "thinking",
        "tool_choice",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
        "betas",
    }
    return {key: value for key, value in body.items() if key in allowed_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one anthropic request from logs_anthropic.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file path, relative to repo root by default. Defaults to .env.",
    )
    parser.add_argument(
        "request_file",
        nargs="?",
        help="Path to a *-req.json file. Defaults to the latest file under logs_anthropic.",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = ROOT_DIR / args.env_file
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    load_dotenv(env_path, override=True)

    proxy_host = os.getenv("PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    proxy_port = (os.getenv("PROXY_PORT", "4000").strip() or "4000")
    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY") or "sk-1234").strip() or "sk-1234"
    verify_ssl = parse_bool(os.getenv("SSL_VERIFY"), default=False)

    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    disable_warnings(InsecureRequestWarning)

    request_path = pick_request_file(args.request_file)
    payload = load_request_payload(request_path)

    http_client = httpx.Client(verify=verify_ssl)
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=f"http://{proxy_host}:{proxy_port}",
        http_client=http_client,
    )

    print(f"env file   : {env_path}")
    print(f"base url   : http://{proxy_host}:{proxy_port}")
    print(f"api key    : {api_key}")
    print(f"request    : {request_path}")
    print(f"model      : {payload.get('model')}")
    print(f"stream     : {payload.get('stream', False)}")
    print()

    response = client.messages.create(**payload)

    if payload.get("stream"):
        for event in response:
            print(event)
    else:
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
