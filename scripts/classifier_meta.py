#!/usr/bin/env python3
"""
classifier_meta.py - Run an LLM classifier over a JSONL file of rows (Meta API).

Usage:
    python classifier_meta.py <input.jsonl> [num_rows] [model] [output.jsonl] [checkpoint_every]

Examples:
    python classifier_meta.py alpaca-output.jsonl
    python classifier_meta.py alpaca-output.jsonl 20
    python classifier_meta.py alpaca-output.jsonl 20 muse-spark-1.3-contributor
    python classify_meta.py apps-output.jsonl 1000 muse-spark-1.3-contributor apps-classified.jsonl 100
"""

import json
import os
import sys
import time
from urllib import request, error

API_URL = "https://api.meta.ai/v1/responses"
API_KEY = os.environ.get("META_API_KEY")

SYSTEM_PROMPT = (
    "Classify the user programming task into exactly one tier:\n"
    "[TIER_1_SIMPLE]: quick syntax, single function, lookup, explanation\n"
    "[TIER_2_MEDIUM]: multi-file changes, bug fixing, medium edits\n"
    "[TIER_3_COMPLEX]: deep architectural reasoning, tricky algorithms, math/threading\n"
    "Answer ONLY with the tag."
)

VALID_TAGS = {"[TIER_1_SIMPLE]", "[TIER_2_MEDIUM]", "[TIER_3_COMPLEX]"}

DEFAULT_CHECKPOINT_EVERY = 100


def extract_text(parsed):
    """Pull assistant text out of either a Responses-style or Chat-style payload."""
    if isinstance(parsed.get("output_text"), str):
        return parsed["output_text"]

    output = parsed.get("output")
    if isinstance(output, list):
        chunks = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        chunks.append(c["text"])
        if chunks:
            return "".join(chunks)

    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        if isinstance(msg.get("content"), str):
            return msg["content"]

    for key in ("completion", "response", "text"):
        if isinstance(parsed.get(key), str):
            return parsed[key]

    return ""


def call_api(model, question, timeout=120):
    """Send one classification request and return the raw assistant text."""
    payload = {
        "reasoning": {"effort": "minimal"},
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
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

    text = extract_text(parsed)
    if not text:
        raise RuntimeError(
            "Could not extract text from response. Raw keys: %s\nRaw body (first 500 chars): %s"
            % (list(parsed.keys()), body[:500])
        )
    return text


def normalize_tag(raw):
    if not raw:
        return "UNKNOWN"
    s = raw.strip().upper()
    for tag in VALID_TAGS:
        if tag in s:
            return tag
    return "UNKNOWN"


def flush_results(output_path, results):
    """Append the pending results to the output file and clear the buffer."""
    if not results:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    results.clear()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    model = sys.argv[3] if len(sys.argv) > 3 else "muse-spark-1.3-contributor"
    output_path = (
        sys.argv[4] if len(sys.argv) > 4
        else input_path.replace(".jsonl", "-classified-meta.jsonl")
    )
    checkpoint_every = (
        int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_CHECKPOINT_EVERY
    )

    if not API_KEY:
        print("ERROR: META_API_KEY environment variable is not set.")
        sys.exit(1)

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows = rows[:max_rows]

    # Determine starting row number if we are resuming (output already exists)
    already_done = 0
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for _ in f:
                already_done += 1
        if already_done:
            print("Resuming: %d row(s) already in %s" % (already_done, output_path))

    # Truncate output if we are starting fresh
    if already_done == 0 and os.path.exists(output_path):
        os.remove(output_path)

    # Skip rows we've already processed
    if already_done:
        rows = rows[already_done:]

    print("Classifying %d row(s) from %s using %s (checkpoint every %d)..."
          % (len(rows), input_path, model, checkpoint_every))

    pending = []
    counts = {}
    total_processed = already_done

    for i, row in enumerate(rows, start=1):
        text = row.get("text", "")
        row_id = row.get("id", "row_%d" % (already_done + i))
        try:
            raw = call_api(model, text)
            tag = normalize_tag(raw)
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            print("  [%d/%d] %s: HTTP %s - %s" % (i, len(rows), row_id, e.code, detail))
            tag = "ERROR"
        except Exception as e:
            print("  [%d/%d] %s: %s: %s" % (i, len(rows), row_id, type(e).__name__, e))
            tag = "ERROR"

        initialLabel = row.get("label", "")
        print("  [%d/%d] %s: %s  (existing: %s)" % (i, len(rows), row_id, tag, initialLabel or "-"))

        record = {
            "id": row_id,
            "text": text,
            "oldlabel": initialLabel,
            "label": tag,
            "model": model,
        }
        pending.append(record)
        counts[tag] = counts.get(tag, 0) + 1
        total_processed += 1

        # Checkpoint flush
        if checkpoint_every > 0 and len(pending) >= checkpoint_every:
            flush_results(output_path, pending)
            print("  [checkpoint] flushed up to %d row(s) -> %s" % (total_processed, output_path))

        time.sleep(0.2)

    # Final flush
    flush_results(output_path, pending)

    print("\nWrote %d row(s) -> %s" % (total_processed, output_path))
    for tag, n in sorted(counts.items(), key=lambda x: -x[1]):
        print("  %s: %d" % (tag, n))


if __name__ == "__main__":
    main()