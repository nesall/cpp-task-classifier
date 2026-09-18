#!/usr/bin/env python3
"""
near_dup.py - Find near-duplicate rows in a JSONL file using a llama.cpp
embedding server.

Usage:
    python near_dup.py <input.jsonl> [threshold] [server_url]

Examples:
    python near_dup.py combined.jsonl
    python near_dup.py combined.jsonl 0.92
    python near_dup.py combined.jsonl 0.90 http://localhost:8583

Outputs:
    - <input>-near-dups.jsonl      clustered rows
    - prints a summary to stdout
"""

import json
import sys
import urllib.request

import numpy as np

DEFAULT_THRESHOLD = 0.90
DEFAULT_SERVER = "http://localhost:8583"
BATCH_SIZE = 32
EMBED_MODEL = "bge-base-en-v1.5-q4_k_m.gguf"  # informational; llama.cpp ignores it


def embed_batch(server, texts):
    """Call llama.cpp /v1/embeddings and return a list of vectors."""
    payload = {"model": EMBED_MODEL, "input": texts}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))

    items = parsed.get("data")
    if not items:
        raise RuntimeError("No 'data' in embedding response: %s" % str(parsed)[:300])

    items = sorted(items, key=lambda x: x["index"])
    return [it["embedding"] for it in items]


def embed_all(server, texts, batch_size=BATCH_SIZE):
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        vecs = embed_batch(server, chunk)
        if len(vecs) != len(chunk):
            raise RuntimeError(
                "Expected %d vectors, got %d (batch at %d)"
                % (len(chunk), len(vecs), start)
            )
        vectors.extend(vecs)
        done = min(start + batch_size, len(texts))
        if done % 320 == 0 or done == len(texts):
            print("  embedded %d/%d" % (done, len(texts)))
    return vectors


def build_similarity(vectors):
    """Return (n, n) cosine similarity matrix as a numpy array."""
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms
    return v @ v.T


def find_clusters(sim, threshold):
    """Union-find over pairs above threshold. Returns dict {root: [indices]}."""
    n = sim.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only upper triangle
    ii, jj = np.where(np.triu(sim, k=1) >= threshold)
    for a, b in zip(ii.tolist(), jj.tolist()):
        union(a, b)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD
    server = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SERVER

    output_path = input_path.replace(".jsonl", "-near-dups.jsonl")

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    texts = [r.get("text", "") for r in rows]
    print("Loaded %d rows from %s" % (len(rows), input_path))
    print("Embedding via %s (batch=%d)..." % (server, BATCH_SIZE))

    vectors = embed_all(server, texts)
    print("  got %d vectors of dim %d" % (len(vectors), len(vectors[0])))

    print("Building similarity matrix...")
    sim = build_similarity(vectors)

    print("Clustering at threshold %.3f..." % threshold)
    clusters = find_clusters(sim, threshold)

    ordered = sorted(clusters.values(), key=lambda v: -len(v))
    total_dupes = sum(len(v) for v in ordered)

    print()
    print("Rows:              %d" % len(rows))
    print("Near-dup clusters: %d" % len(ordered))
    print("Rows in clusters:  %d (%.1f%%)"
          % (total_dupes, 100.0 * total_dupes / len(rows) if rows else 0.0))
    print()

    for ci, cluster in enumerate(ordered[:10], 1):
        print("Cluster %d (%d rows):" % (ci, len(cluster)))
        for idx in cluster[:3]:
            print("  [%d] %s" % (idx, texts[idx][:110]))
        if len(cluster) > 3:
            print("  ... and %d more" % (len(cluster) - 3))
        print()

    with open(output_path, "w", encoding="utf-8") as f:
        for ci, cluster in enumerate(ordered, 1):
            for idx in cluster:
                f.write(json.dumps({
                    "cluster": ci,
                    "cluster_size": len(cluster),
                    "index": idx,
                    "id": rows[idx].get("id", ""),
                    "label": rows[idx].get("label", ""),
                    "text": texts[idx],
                }, ensure_ascii=False) + "\n")

    print("Wrote %d clustered rows -> %s" % (total_dupes, output_path))


if __name__ == "__main__":
    main()