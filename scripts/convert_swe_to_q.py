#!/usr/bin/env python3
"""
Extract natural-language C++ programming questions from SWE-bench problem statements
using an LLM API.

Usage:
    python extract_questions.py input.jsonl -o output.jsonl
    python extract_questions.py input.jsonl -o output.jsonl \
        --url https://api.meta.ai/v1/responses \
        --model muse-spark-1.3-contributor

Requires env var META_API_KEY (or pass --api-key).
"""

import argparse
import json
import os
import sys
import time
import hashlib
import random
from typing import Optional

import urllib.request
import urllib.error

SYSTEM_PROMPT = (
    "You are an expert software engineer. Read the following software engineering "
    "issue. Extract the core architectural, concurrency, or system design problem.\n"
    "Rewrite it as a single, concise query (1 to 3 sentences) asked by a developer "
    "seeking AI assistance.\n\n"
    "CRITICAL RULES:\n\n"
    "Strip out references to the original frameworks and stack traces.\n\n"
    "Pick one modern language from this list and write the query as a native speaker "
    "of that language would: C++, Rust, Go, Java, Python, TypeScript.\n"
    "Express the language choice ONLY through idioms, standard-library types, and "
    "ecosystem vocabulary native to that language. NEVER name the language explicitly "
    "and NEVER start the sentence with 'In <language>'.\n\n"
    "VARY THE OPENING. Do NOT begin with 'How do I', 'How can I', 'How should I', or "
    "'What is the best way to'. Do NOT begin with 'In <language>'. Choose a different, "
    "natural opening each time. Prefer one of these forms:\n"
    "- A declarative problem statement (e.g., 'My worker pool leaks tasks when...')\n"
    "- An imperative request (e.g., 'Explain how to keep ... alive across ...')\n"
    "- A noun-phrase prompt (e.g., 'Design pattern for ... when ...')\n"
    "- A 'Why does ...' or 'What happens if ...' question\n"
    "- A 'Looking for ...' / 'Need help with ...' framing\n\n"
    "Do not include markdown, logs, or code blocks in your output. Just the natural "
    "language query."
)

DEFAULT_URL = "https://api.meta.ai/v1/responses"
DEFAULT_MODEL = "muse-spark-1.3-contributor"

PROMPT_VERSION = "v0"

LANGS = ["C++", "Rust", "Go", "Java", "Python", "TypeScript"]
OPENING_STYLES = [
    "a declarative problem statement, e.g. 'My X leaks Y when Z'",
    "an imperative request, e.g. 'Explain how to ...', 'Show the idiomatic way to ...'",
    "a noun-phrase prompt, e.g. 'Design pattern for ... when ...'",
    "a 'Why does ...' or 'What happens if ...' question",
    "a 'Looking for ...' / 'Need help with ...' framing",
    "a debugging-style statement, e.g. 'This code drops state when ...'",
]


def _is_chat_completions(url: str) -> bool:
    """Heuristic: OpenAI-compatible servers (llama-server, vLLM, etc.) use
    a /chat/completions path; Meta's Responses API does not."""
    return "/chat/completions" in url


def call_llm(problem_statement: str, url: str, model: str, api_key: str,
             max_retries: int = 3, timeout: int = 120) -> str:
    """Send a single problem statement to the LLM and return the generated text."""
    chosen_lang = random.choice(LANGS)
    style = random.choice(OPENING_STYLES)
    user_content = (
        f"Rewrite this issue as a {chosen_lang} developer's question. "
        f"Use only native {chosen_lang} idioms and do NOT name the language.\n"
        f"Begin the query as {style}. "
        f"Do NOT begin with 'How do I', 'How can I', or 'In {chosen_lang}'.\n\n"
        f"{problem_statement}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if _is_chat_completions(url):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.3,
        }
    else:
        payload = {
            "reasoning": {"effort": "minimal"},
            "model": model,
            "input": messages,
        }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "unused":
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return _extract_text(parsed)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = 2 ** attempt
                print(f"  [retry {attempt}/{max_retries}] {e}; sleeping {sleep_s}s",
                      file=sys.stderr)
                time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def _extract_text(parsed: dict) -> str:
    """
    Extract generated text from a Responses-API style payload.
    Handles a few common shapes so we're not overly coupled to one schema.
    """
    # Shape 1: {"output_text": "..."}
    if isinstance(parsed.get("output_text"), str):
        return parsed["output_text"].strip()

    # Shape 2: {"output": [{"content": [{"type": "output_text", "text": "..."}]}]}
    output = parsed.get("output")
    if isinstance(output, list):
        chunks = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        chunks.append(c["text"])
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "".join(chunks).strip()

    # Shape 3: OpenAI chat-completions fallback
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        if isinstance(msg.get("content"), str):
            return msg["content"].strip()

    raise RuntimeError(f"Could not locate text in LLM response: {parsed!r}")

def key_for(ps: str) -> str:
    return hashlib.sha256((PROMPT_VERSION + "\x00" + ps).encode("utf-8")).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input JSONL file (one object per line with "
                                  "'problem_statement')")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL file")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"LLM endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--api-key", default=None,
                    help="API key (defaults to $META_API_KEY)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to sleep between requests (rate limiting)")
    ap.add_argument("--skip-errors", action="store_true",
                    help="Continue on per-record failure instead of aborting")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("META_API_KEY")
    if not api_key and not _is_chat_completions(args.url):
        print("error: META_API_KEY not set and --api-key not provided", file=sys.stderr)
        return 2
        
        
    cache = {}  # hash -> {"text": ..., "label": ...}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "hash" in obj:
                    cache[obj["hash"]] = obj    

    # Read all input rows first so we can resume/append cleanly if needed.
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {line_no}: {e}",
                      file=sys.stderr)
                continue
            ps = obj.get("problem_statement")
            if not isinstance(ps, str) or not ps.strip():
                print(f"warning: line {line_no} has no 'problem_statement'",
                      file=sys.stderr)
                continue
            records.append(ps)

    print(f"Loaded {len(records)} problem statements.", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as out:
        for idx, ps in enumerate(records, 1):
            print(f"[{idx}/{len(records)}] querying LLM...", file=sys.stderr)
            h = key_for(ps)
            if h in cache:
                out.write(json.dumps(cache[h], ensure_ascii=False) + "\n")
                continue
            try:
                text = call_llm(ps, args.url, args.model, api_key)
            except Exception as e:
                if args.skip_errors:
                    print(f"  failed: {e}", file=sys.stderr)
                    continue
                raise
            out.write(json.dumps({"hash": h, "text": text, "label": ""}, ensure_ascii=False) + "\n")
            out.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())