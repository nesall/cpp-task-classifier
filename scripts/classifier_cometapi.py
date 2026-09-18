#!/usr/bin/env python3
"""
classify.py - Run an LLM classifier over a JSONL file of rows.

Usage:
    python classify.py <input.jsonl> [num_rows] [model] [output.jsonl]

Examples:
    python classify.py alpaca-output.jsonl
    python classify.py alpaca-output.jsonl 20
    python classify.py alpaca-output.jsonl 20 gpt-5-nano
    python classify.py alpaca-output.jsonl 20 gpt-5-nano alpaca-classified.jsonl
"""

import json
import os
import sys
import time
from urllib import request, error

API_URL = "https://api.cometapi.com/v1/chat/completions"
API_KEY = os.environ.get("COMET_API_KEY")

SYSTEM_PROMPT = (
    "Classify the user programming task into exactly one tier:\n"
    "[TIER_1_SIMPLE]: quick syntax, single function, lookup, explanation\n"
    "[TIER_2_MEDIUM]: multi-file changes, bug fixing, medium edits\n"
    "[TIER_3_COMPLEX]: deep architectural reasoning, tricky algorithms, math/threading\n"
    "Answer ONLY with the tag."
)

VALID_TAGS = {"[TIER_1_SIMPLE]", "[TIER_2_MEDIUM]", "[TIER_3_COMPLEX]"}


def call_api(model, question, timeout=60):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def normalize_tag(raw):
    if not raw:
        return "N/A"
    s = raw.strip().upper()
    for tag in VALID_TAGS:
        if tag in s:
            return tag
    return tag


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5-nano"
    output_path = (
        sys.argv[4] if len(sys.argv) > 4
        else input_path.replace(".jsonl", "-classified.jsonl")
    )

    if not API_KEY:
        print("ERROR: COMETAPI_KEY environment variable is not set.")
        sys.exit(1)

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows = rows[:max_rows]
    print("Classifying %d row(s) from %s using %s..." % (len(rows), input_path, model))

    results = []
    for i, row in enumerate(rows, start=1):
        text = row.get("text", "")
        row_id = row.get("id", "row_%d" % i)
        try:
            raw = call_api(model, text)
            tag = normalize_tag(raw)
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            print("  [%d/%d] %s: HTTP %s - %s" % (i, len(rows), row_id, e.code, detail))
            tag = "ERROR"
        except Exception as e:
            print("  [%d/%d] %s: %s: %s" % (i, len(rows), row_id, type(e).__name__, e))
            tag = "ERROR"

        predicted = row.get("label", "")
        print("  [%d/%d] %s: %s  (existing: %s)" % (i, len(rows), row_id, tag, predicted or "-"))
        results.append({
            "id": row_id,
            "text": text,
            "label": predicted,
            "predicted": tag,
            "model": model,
        })
        time.sleep(0.2)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in results:
        counts[r["predicted"]] = counts.get(r["predicted"], 0) + 1
    print("\nWrote %d row(s) -> %s" % (len(results), output_path))
    for tag, n in sorted(counts.items(), key=lambda x: -x[1]):
        print("  %s: %d" % (tag, n))


if __name__ == "__main__":
    main()