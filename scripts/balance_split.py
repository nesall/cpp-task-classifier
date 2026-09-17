#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

input_file = Path("data/querying/combined_all.jsonl")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

data_by_tier = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row["label"].strip("[]")
        if label == "TIER_2_REFACTOR":
            label = "TIER_2_MEDIUM"
        row["label"] = label
        data_by_tier[label].append(row)

# Option A: Strict balance (downsample Tier 1 to ~1,244)
# Option B: Cap Tier 1 at 1,800 to preserve data while limiting bias
min_samples = min(len(v) for v in data_by_tier.values())
target_per_tier = 1500  # Cap Tier 1, keep all Tier 2 & Tier 3

balanced_pool = []
for label, items in data_by_tier.items():
    random.shuffle(items)
    selected = items[:target_per_tier] if len(items) > target_per_tier else items
    balanced_pool.extend(selected)

random.shuffle(balanced_pool)

n = len(balanced_pool)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train = balanced_pool[:n_train]
val = balanced_pool[n_train:n_train + n_val]
test = balanced_pool[n_train + n_val:]

for name, subset in [("train", train), ("val", val), ("test", test)]:
    with open(output_dir / f"{name}.jsonl", "w", encoding="utf-8") as out:
        for item in subset:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {name}.jsonl: {len(subset)} items")