"""Live single-signature smoke test for the Reflection consumer.

Usage:
    # Through local proxy (default endpoint http://127.0.0.1:4001):
    REFLECTION_API_KEY=sk-xxx python3 test/test_reflection_consumer_live.py

    # Direct to Anthropic API:
    REFLECTION_API_KEY=sk-xxx REFLECTION_ENDPOINT=https://api.anthropic.com \
        REFLECTION_MODEL=claude-opus-4-8 python3 test/test_reflection_consumer_live.py

Environment variables:
    REFLECTION_API_KEY   - API key (required)
    REFLECTION_ENDPOINT  - base URL (default: http://127.0.0.1:4001)
    REFLECTION_MODEL     - model name (default: THINKING:claude-opus-4-8)
    REFLECTION_TIMEOUT   - seconds (default: 600)
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.thinking_reflection.config import load_config
from src.thinking_reflection.consumer import reflect
from src.thinking_reflection.prompt_loader import load_prompt


SIGNATURE_SAMPLE = Path(
    "/home/nianzuzheng/project/trajectory_qc/signature_reflection_bulk/"
    "src/raw/signature_sample.txt"
)

DEFAULT_ENDPOINT = "http://127.0.0.1:4001"
DEFAULT_MODEL = "THINKING:claude-opus-4-8"
DEFAULT_TIMEOUT = 600
REFLECTION_API_KEY = "sk-b64c8203"


class ReflectionSingleTest(unittest.TestCase):
    """Direct call to reflect() — no proxy login needed, just API key + endpoint."""

    def test_reflect_bulk(self) -> None:
        """Single signature bulk reflection test."""
        api_key = REFLECTION_API_KEY
        if not api_key:
            self.skipTest("REFLECTION_API_KEY not set")

        endpoint = os.environ.get("REFLECTION_ENDPOINT", DEFAULT_ENDPOINT).strip()
        model = os.environ.get("REFLECTION_MODEL", DEFAULT_MODEL).strip()
        timeout = int(os.environ.get("REFLECTION_TIMEOUT", DEFAULT_TIMEOUT))

        self.assertTrue(SIGNATURE_SAMPLE.exists(), f"Missing: {SIGNATURE_SAMPLE}")
        signature = SIGNATURE_SAMPLE.read_text(encoding="utf-8").strip()
        self.assertTrue(signature, "signature fixture must not be empty")

        prompt = load_prompt(load_config().prompt_dir, "bulk")

        print(f"\n--- Reflection Single Test ---")
        print(f"endpoint : {endpoint}")
        print(f"model    : {model}")
        print(f"method   : bulk")
        print(f"timeout  : {timeout}s")
        print(f"sig len  : {len(signature)} chars")
        print(f"Sending request...")

        result = reflect(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            instruction=prompt.instruction,
            tool=prompt.tool,
            unrelated_thinking=prompt.unrelated_thinking,
            signature=signature,
            thinking=None,
            method="bulk",
            stream=False,
            max_tokens=16384,
            timeout=timeout,
        )

        print(f"\n--- Result ---")
        print(f"response_id  : {result.response_id}")
        print(f"model        : {result.model}")
        print(f"stop_reason  : {result.stop_reason}")
        print(f"sentence_cnt : {result.sentence_count}")
        if result.usage:
            print(f"tokens       : input={result.usage.get('input_tokens')} output={result.usage.get('output_tokens')}")
        print(f"output_len   : {len(result.text)} chars")
        print(f"\n--- Output Preview (first 800 chars) ---")
        print(result.text[:800])

        self.assertTrue(result.text.strip(), "output text must not be empty")
        self.assertEqual(result.sentence_count, 1)
        self.assertTrue(result.response_id)
        print("\n--- PASS ---")


if __name__ == "__main__":
    unittest.main(verbosity=2)
