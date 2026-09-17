#!/usr/bin/env python3
"""
Dataset Reconciliation and Stratified Splitting Script.
Cleans label poisoning, drops high-similarity near-duplicates, and outputs balanced splits.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd

random.seed(42)

DATA_DIR = Path("data")
QUERY_DIR = DATA_DIR / "querying"
REPORT_DIR = QUERY_DIR / "quality_report"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load combined_all.jsonl
combined_file = QUERY_DIR / "combined_all.jsonl"
if not combined_file.exists():
    # Fallback to local root or data/ if path differs
    combined_file = Path("combined_all.jsonl")

print(f"Loading raw dataset from: {combined_file}")
records = []
with open(combined_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row["label"].strip("[]")
        if label == "TIER_2_REFACTOR":
            label = "TIER_2_MEDIUM"
        records.append({
            "id": idx,
            "text": row["text"],
            "label": label
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} total raw samples.")

# 2. Reconcile Cross-Label Pairs using cross_label_similar_pairs.csv
cross_file = REPORT_DIR / "cross_label_similar_pairs.csv"
if cross_file.exists():
    cross_df = pd.read_csv(cross_file)
    print(f"Processing {len(cross_df)} cross-label collision pairs...")

    drop_ids = set()
    concurrency_re = re.compile(
        r"(lock-free|concurren|atomic|mutex|deadlock|race condition|compare-and-swap|spurious wakeup|hazard pointer|memory_order|thread|park)",
        re.IGNORECASE
    )

    for _, row in cross_df.iterrows():
        id_a, id_b = int(row["id_a"]), int(row["id_b"])
        text_a, text_b = str(row["text_a"]), str(row["text_b"])
        lbl_a, lbl_b = str(row["label_a"]).strip("[]"), str(row["label_b"]).strip("[]")

        # Case 1: T2 vs T3 Concurrency conflict -> Must be TIER_3_COMPLEX
        if {lbl_a, lbl_b} == {"TIER_2_MEDIUM", "TIER_3_COMPLEX"}:
            t3_id = id_a if lbl_a == "TIER_3_COMPLEX" else id_b
            t2_id = id_b if lbl_a == "TIER_3_COMPLEX" else id_a

            if concurrency_re.search(text_a) or concurrency_re.search(text_b):
                # Ensure the retained sample is TIER_3_COMPLEX, drop the T2 clone
                df.loc[df["id"] == t3_id, "label"] = "TIER_3_COMPLEX"
                drop_ids.add(t2_id)
            else:
                # Generic structural refactor clone -> drop duplicate
                drop_ids.add(id_b)

        # Case 2: Exact identical text cross-label (e.g., Doxygen prompt)
        elif text_a.strip().lower() == text_b.strip().lower():
            drop_ids.add(id_b)

    df = df[~df["id"].isin(drop_ids)].copy()
    print(f"Removed {len(drop_ids)} poisoned/conflicting duplicate rows. Retained: {len(df)} rows.")

# 3. Deduplicate High-Similarity Pairs (>= 0.96)
sim_file = REPORT_DIR / "similar_pairs.csv"
if sim_file.exists():
    sim_df = pd.read_csv(sim_file)
    high_sim = sim_df[sim_df["similarity"] >= 0.96]
    print(f"Deduplicating {len(high_sim)} near-identical pairs (similarity >= 0.96)...")

    sim_drop_ids = set()
    for _, row in high_sim.iterrows():
        id_a, id_b = int(row["id_a"]), int(row["id_b"])
        if id_a not in sim_drop_ids and id_b in df["id"].values:
            sim_drop_ids.add(id_b)

    df = df[~df["id"].isin(sim_drop_ids)].copy()
    print(f"Removed {len(sim_drop_ids)} high-similarity near-clones. Retained: {len(df)} rows.")

# 4. Final Class Distribution Check
print("\nReconciled Class Distribution:")
counts = df["label"].value_counts()
for lbl, c in counts.items():
    print(f"  {lbl:20s}: {c:5d} ({100.0 * c / len(df):.1f}%)")

# 5. Balance Classes and Create 70 / 15 / 15 Splits
# Determine per-class sample size (cap to avoid severe imbalance)
min_class = counts.min()
target_per_class = min(min_class, 1600)  # Use up to 1,600 samples per class
print(f"\nSampling {target_per_class} per class for a balanced dataset of {target_per_class * 3} samples...")

balanced_dfs = []
for lbl in ["TIER_1_SIMPLE", "TIER_2_MEDIUM", "TIER_3_COMPLEX"]:
    tier_subset = df[df["label"] == lbl].sample(n=target_per_class, random_state=42)
    balanced_dfs.append(tier_subset)

clean_pool = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

total_n = len(clean_pool)
n_train = int(total_n * 0.70)
n_val = int(total_n * 0.15)

train_df = clean_pool.iloc[:n_train]
val_df = clean_pool.iloc[n_train:n_train + n_val]
test_df = clean_pool.iloc[n_train + n_val:]

for split_name, split_data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out_path = OUT_DIR / f"{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in split_data.iterrows():
            item = {
                "id": f"rec_{int(r['id']):05d}",
                "text": str(r["text"]),
                "label": str(r["label"])
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(split_data)} samples)")

print("\nReconciliation complete.")