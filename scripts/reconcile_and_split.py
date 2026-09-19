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

combined_file = QUERY_DIR / "combined_all_v2.jsonl"
if not combined_file.exists():
    # Fallback to local root or data/ if path differs
    combined_file = Path("combined_all_v2.jsonl")

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
        r"(lock|unlock|concurren|atomic|mutex|deadlock|race|races|compare-and-swap|spurious wakeup|hazard pointer|memory_order|thread|threads|park|spin|retry loop)",
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

# Filter out any malformed / ERROR labels first
valid_labels = {"TIER_1_SIMPLE", "TIER_2_MEDIUM", "TIER_3_COMPLEX"}
df = df[df["label"].isin(valid_labels)].copy()

# 4. Final Class Distribution Check
print("\nReconciled Class Distribution:")
counts = df["label"].value_counts()
for lbl, c in counts.items():
    print(f"  {lbl:20s}: {c:5d} ({100.0 * c / len(df):.1f}%)")

# 5. Balance Classes and Create 70 / 15 / 15 Splits
# TIER_2_MEDIUM is the bottleneck at 1,741 samples
#target_per_class = min(counts.min(), 1740)
#print(f"\nSampling {target_per_class} per class for a balanced dataset of {target_per_class * 3} samples...")
#balanced_dfs = []
#for lbl in ["TIER_1_SIMPLE", "TIER_2_MEDIUM", "TIER_3_COMPLEX"]:
#    tier_subset = df[df["label"] == lbl].sample(n=target_per_class, random_state=42)
#    balanced_dfs.append(tier_subset)

#clean_pool = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

# 5. Use full reconciled dataset (no undersampling) - stratified 70/15/15 split
clean_pool = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"\nUsing full reconciled dataset: {len(clean_pool)} samples (no class balancing).")

total_n = len(clean_pool)
n_train = int(total_n * 0.70)
n_val = int(total_n * 0.15)

#train_df = clean_pool.iloc[:n_train]
#val_df = clean_pool.iloc[n_train:n_train + n_val]
#test_df = clean_pool.iloc[n_train + n_val:]
train_dfs, val_dfs, test_dfs = [], [], []
for lbl in ["TIER_1_SIMPLE", "TIER_2_MEDIUM", "TIER_3_COMPLEX"]:
    subset = clean_pool[clean_pool["label"] == lbl].sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(subset)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)
    train_dfs.append(subset.iloc[:n_tr])
    val_dfs.append(subset.iloc[n_tr:n_tr + n_va])
    test_dfs.append(subset.iloc[n_tr + n_va:])

train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

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