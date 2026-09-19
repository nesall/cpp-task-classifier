import json, random, csv, sys

def sample_source(input_path, source_name, n=40, seed=42):
    with open(input_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    # Stratify by label so you see a mix of tiers, not whatever's most common
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    sample = []
    per_label = max(1, n // max(1, len(by_label)))
    for label, items in by_label.items():
        random.shuffle(items)
        sample.extend(items[:per_label])

    random.shuffle(sample)
    sample = sample[:n]

    out_path = f"gold_sample_{source_name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "text", "model_label", "human_label", "notes"])
        for i, r in enumerate(sample):
            w.writerow([f"{source_name}_{i}", source_name, r["text"], r["label"], "", ""])

    print(f"Wrote {len(sample)} rows to {out_path}")

if __name__ == "__main__":
    # Usage: python sample_gold.py <input.jsonl> <source_name> [n]
    input_path = sys.argv[1]
    source_name = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    sample_source(input_path, source_name, n)
