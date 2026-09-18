#!/usr/bin/env python3

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_URL = "http://localhost:8584/v1/embeddings"
DEFAULT_MODEL = "bge-m3"

DEFAULT_BATCH_SIZE = 10
DEFAULT_CHECKPOINT = 100
DEFAULT_TOP_K = 10

REQUEST_TIMEOUT = 300
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: invalid JSON on line {line_no}: {e}")
                continue

            if "text" not in obj or "label" not in obj:
                print(
                    f"WARNING: missing text/label "
                    f"on line {line_no}"
                )
                continue

            rows.append({
                "id": len(rows),
                "text": str(obj["text"]),
                "label": str(obj["label"]),
                "normalized": normalize_text(str(obj["text"])),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exact duplicates
# ---------------------------------------------------------------------------

def find_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, group in df.groupby("normalized"):
        if len(group) < 2:
            continue

        indices = group.index.tolist()

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = df.loc[indices[i]]
                b = df.loc[indices[j]]

                records.append({
                    "id_a": int(a["id"]),
                    "id_b": int(b["id"]),
                    "label_a": a["label"],
                    "label_b": b["label"],
                    "similarity": 1.0,
                    "text_a": a["text"],
                    "text_b": b["text"],
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# llama.cpp embedding API
# ---------------------------------------------------------------------------

def request_embeddings(
    session: requests.Session,
    url: str,
    model: str,
    texts: list[str],
) -> np.ndarray:

    payload = {
        "model": model,
        "input": texts,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                url,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            if "data" not in data:
                raise RuntimeError(
                    f"llama.cpp response has no 'data': {data}"
                )

            items = data["data"]

            if len(items) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings, "
                    f"got {len(items)}"
                )

            # The API normally includes an "index".
            # Sort by it when available to guarantee ordering.
            if all("index" in item for item in items):
                items = sorted(
                    items,
                    key=lambda x: x["index"],
                )

            embeddings = np.asarray(
                [item["embedding"] for item in items],
                dtype=np.float32,
            )

            if embeddings.ndim != 2:
                raise RuntimeError(
                    f"Unexpected embedding shape: "
                    f"{embeddings.shape}"
                )

            return embeddings

        except Exception as e:
            last_error = e

            print(
                f"\nEmbedding request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Embedding request failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------

def save_state(
    state_file: Path,
    completed: int,
    total: int,
    embedding_dim: int,
):
    state = {
        "completed": completed,
        "total": total,
        "embedding_dim": embedding_dim,
    }

    tmp = state_file.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    tmp.replace(state_file)


def load_state(state_file: Path):
    if not state_file.exists():
        return None

    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Embedding generation / resume
# ---------------------------------------------------------------------------

def generate_embeddings(
    df: pd.DataFrame,
    url: str,
    model: str,
    output_dir: Path,
    batch_size: int,
    checkpoint_interval: int,
) -> np.ndarray:

    n = len(df)

    embeddings_file = output_dir / "embeddings.npy"
    checkpoint_file = output_dir / "embeddings_checkpoint.npy"
    state_file = output_dir / "embedding_state.json"

    state = load_state(state_file)

    completed = 0
    embedding_dim = None

    # -------------------------------------------------------
    # Resume existing checkpoint
    # -------------------------------------------------------

    if (
        state is not None
        and checkpoint_file.exists()
    ):
        completed = int(state["completed"])
        embedding_dim = int(state["embedding_dim"])

        print()
        print(
            f"Found checkpoint: "
            f"{completed}/{n} samples"
        )

        checkpoint = np.load(checkpoint_file)

        # Checkpoint holds only the completed prefix.
        if checkpoint.ndim != 2:
            raise RuntimeError(
                f"Unexpected checkpoint shape: {checkpoint.shape}"
            )

        if checkpoint.shape[1] != embedding_dim:
            raise RuntimeError(
                "Checkpoint embedding dimension doesn't match "
                f"state: {checkpoint.shape} vs {embedding_dim}"
            )

        if checkpoint.shape[0] != completed:
            raise RuntimeError(
                "Checkpoint row count doesn't match state: "
                f"{checkpoint.shape[0]} vs {completed}"
            )

        if completed > n:
            raise RuntimeError(
                f"Checkpoint has {completed} samples but "
                f"dataset only has {n}. Dataset shrank?"
            )

        # Allocate full-size array and copy the prefix.
        embeddings = np.zeros(
            (n, embedding_dim),
            dtype=np.float32,
        )
        embeddings[:completed] = checkpoint

    else:
        print()
        print("No embedding checkpoint found.")
        print("Starting from sample 0.")

        embeddings = None

    # -------------------------------------------------------
    # Already complete
    # -------------------------------------------------------

    if completed >= n:
        print("Embeddings already complete.")

        if not embeddings_file.exists():
            np.save(embeddings_file, embeddings)

        return embeddings

    # -------------------------------------------------------
    # Embedding generation
    # -------------------------------------------------------

    session = requests.Session()

    start_time = time.time()

    while completed < n:

        batch_end = min(
            completed + batch_size,
            n,
        )

        texts = df.iloc[
            completed:batch_end
        ]["text"].tolist()

        print(
            f"\rEmbedding "
            f"{completed + 1}-{batch_end}/{n}",
            end="",
            flush=True,
        )

        batch_embeddings = request_embeddings(
            session,
            url,
            model,
            texts,
        )

        # First batch determines dimensionality.
        if embeddings is None:
            embedding_dim = batch_embeddings.shape[1]

            embeddings = np.zeros(
                (n, embedding_dim),
                dtype=np.float32,
            )

        if batch_embeddings.shape[1] != embedding_dim:
            raise RuntimeError(
                "Embedding dimension changed during processing: "
                f"expected {embedding_dim}, "
                f"got {batch_embeddings.shape[1]}"
            )

        embeddings[
            completed:batch_end
        ] = batch_embeddings

        completed = batch_end

        # ---------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------

        should_checkpoint = (
            completed % checkpoint_interval == 0
            or completed == n
        )

        if should_checkpoint:
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0

            remaining = (
                n - completed
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print()

            print(
                f"Checkpoint: {completed}/{n} "
                f"({100.0 * completed / n:.1f}%)"
            )
            
            print(f"Elapsed: {elapsed / 60:.1f} minutes")

            print(
                f"Speed: {rate:.2f} samples/sec"
            )

            print(
                f"ETA: {eta / 60:.1f} minutes"
            )

            # Atomic-ish checkpoint:
            tmp = checkpoint_file.with_suffix(".tmp.npy")

            np.save(tmp, embeddings)

            tmp.replace(checkpoint_file)

            save_state(
                state_file,
                completed,
                n,
                embedding_dim,
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_file}"
            )

    # -------------------------------------------------------
    # Final embedding file
    # -------------------------------------------------------

    np.save(
        embeddings_file,
        embeddings,
    )

    print()
    print(
        f"Final embeddings saved: "
        f"{embeddings_file}"
    )

    return embeddings


# ---------------------------------------------------------------------------
# Normalize embeddings
# ---------------------------------------------------------------------------

def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(norms, 1e-12)

    return embeddings / norms


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def compute_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:

    print()
    print("Computing pairwise cosine similarities...")

    similarity = embeddings @ embeddings.T

    similarity = np.clip(
        similarity,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        similarity,
        -1.0,
    )

    return similarity


# ---------------------------------------------------------------------------
# Similarity distribution
# ---------------------------------------------------------------------------

def print_similarity_stats(
    similarity: np.ndarray,
):
    n = similarity.shape[0]

    values = similarity[
        np.triu_indices(n, k=1)
    ]

    print()
    print("Similarity distribution")
    print("-----------------------")

    for percentile in [
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]:
        print(
            f"{percentile:5.1f}% : "
            f"{np.percentile(values, percentile):.4f}"
        )

    print()
    print("Threshold counts")
    print("----------------")

    for threshold in [
        0.80,
        0.85,
        0.90,
        0.92,
        0.95,
        0.97,
        0.98,
        0.99,
    ]:
        count = int(
            np.sum(values >= threshold)
        )

        print(
            f">= {threshold:.2f}: "
            f"{count:,} pairs"
        )


# ---------------------------------------------------------------------------
# Nearest neighbors
# ---------------------------------------------------------------------------

def generate_nearest_neighbors(
    df: pd.DataFrame,
    similarity: np.ndarray,
    top_k: int,
) -> pd.DataFrame:

    records = []

    for i in range(len(df)):

        indices = np.argpartition(
            similarity[i],
            -top_k,
        )[-top_k:]

        indices = indices[
            np.argsort(
                similarity[i][indices]
            )[::-1]
        ]

        for rank, j in enumerate(indices, 1):

            records.append({
                "id": i,
                "rank": rank,
                "neighbor_id": int(j),
                "similarity": float(
                    similarity[i, j]
                ),
                "label": df.iloc[i]["label"],
                "neighbor_label": (
                    df.iloc[j]["label"]
                ),
                "text": df.iloc[i]["text"],
                "neighbor_text": (
                    df.iloc[j]["text"]
                ),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# High similarity pairs
# ---------------------------------------------------------------------------

def generate_pairs(
    df: pd.DataFrame,
    similarity: np.ndarray,
    threshold: float,
) -> pd.DataFrame:

    print()
    print(
        f"Finding pairs with similarity >= "
        f"{threshold:.3f}..."
    )

    records = []

    n = len(df)

    for i in range(n):

        matches = np.where(
            similarity[i, i + 1:] >= threshold
        )[0]

        for offset in matches:

            j = i + 1 + int(offset)

            records.append({
                "id_a": i,
                "id_b": j,
                "similarity": float(
                    similarity[i, j]
                ),
                "label_a": df.iloc[i]["label"],
                "label_b": df.iloc[j]["label"],
                "same_label": (
                    df.iloc[i]["label"]
                    ==
                    df.iloc[j]["label"]
                ),
                "text_a": df.iloc[i]["text"],
                "text_b": df.iloc[j]["text"],
            })

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            "similarity",
            ascending=False,
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze semantic similarity in "
            "a JSONL classification dataset."
        )
    )

    parser.add_argument(
        "dataset",
        help="Input JSONL file",
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            f"llama.cpp embeddings URL "
            f"(default: {DEFAULT_URL})"
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Embedding model "
            f"(default: {DEFAULT_MODEL})"
        ),
    )

    parser.add_argument(
        "--output",
        default="quality_report",
        help=(
            "Output directory "
            "(default: quality_report)"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Embedding batch size "
            f"(default: {DEFAULT_BATCH_SIZE})"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=DEFAULT_CHECKPOINT,
        help=(
            f"Checkpoint interval "
            f"(default: {DEFAULT_CHECKPOINT})"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help=(
            "Near-duplicate similarity threshold "
            "(default: 0.90)"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Nearest neighbors per sample "
            f"(default: {DEFAULT_TOP_K})"
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.checkpoint < 1:
        raise ValueError(
            "--checkpoint must be >= 1"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = Path(args.dataset)

    # -------------------------------------------------------
    # Dataset
    # -------------------------------------------------------

    print(
        f"Loading dataset: "
        f"{dataset_path}"
    )

    df = load_dataset(dataset_path)

    print()
    print("Dataset")
    print("-------")

    print(
        f"Samples: {len(df):,}"
    )

    print()
    print("Labels:")

    for label, count in (
        df["label"].value_counts().items()
    ):
        percentage = (
            100.0 * count / len(df)
        )

        print(
            f"  {label:20s} "
            f"{count:5d} "
            f"({percentage:5.1f}%)"
        )

    unique = df["normalized"].nunique()

    print()
    print(
        f"Unique normalized questions: "
        f"{unique:,}"
    )

    print(
        f"Exact duplicate samples: "
        f"{len(df) - unique:,}"
    )

    # -------------------------------------------------------
    # Exact duplicates
    # -------------------------------------------------------

    exact = find_exact_duplicates(df)

    exact_path = (
        output_dir /
        "exact_duplicates.csv"
    )

    exact.to_csv(
        exact_path,
        index=False,
    )

    print()
    print(
        f"Exact duplicate pairs: "
        f"{len(exact):,}"
    )

    # -------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------

    print()
    print("Embedding configuration")
    print("-----------------------")

    print(f"Server:       {args.url}")
    print(f"Model:        {args.model}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Checkpoint:   {args.checkpoint}")

    embeddings = generate_embeddings(
        df=df,
        url=args.url,
        model=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint,
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # -------------------------------------------------------
    # Normalize
    # -------------------------------------------------------

    embeddings = normalize_embeddings(
        embeddings
    )

    # -------------------------------------------------------
    # Similarity
    # -------------------------------------------------------

    similarity = compute_similarity_matrix(
        embeddings
    )

    print_similarity_stats(
        similarity
    )

    # -------------------------------------------------------
    # Nearest neighbors
    # -------------------------------------------------------

    neighbors = generate_nearest_neighbors(
        df,
        similarity,
        args.top_k,
    )

    neighbors_path = (
        output_dir /
        "nearest_neighbors.csv"
    )

    neighbors.to_csv(
        neighbors_path,
        index=False,
    )

    print()
    print(
        f"Nearest neighbors saved: "
        f"{neighbors_path}"
    )

    # -------------------------------------------------------
    # Similar pairs
    # -------------------------------------------------------

    pairs = generate_pairs(
        df,
        similarity,
        args.threshold,
    )

    pairs_path = (
        output_dir /
        "similar_pairs.csv"
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    print(
        f"Similar pairs saved: "
        f"{pairs_path}"
    )

    # -------------------------------------------------------
    # Cross-label pairs
    # -------------------------------------------------------

    if not pairs.empty:

        cross_label = pairs[
            pairs["label_a"]
            !=
            pairs["label_b"]
        ].copy()

        cross_label_path = (
            output_dir /
            "cross_label_similar_pairs.csv"
        )

        cross_label.to_csv(
            cross_label_path,
            index=False,
        )

        print()
        print(
            f"Cross-label similar pairs: "
            f"{len(cross_label):,}"
        )

        print(
            f"Saved: "
            f"{cross_label_path}"
        )

        print()
        print(
            "Top cross-label pairs"
        )
        print(
            "----------------------"
        )

        for _, row in (
            cross_label.head(20).iterrows()
        ):

            print()
            print(
                f"Similarity: "
                f"{row['similarity']:.4f}"
            )

            print(
                f"A [{row['label_a']}]: "
                f"{row['text_a']}"
            )

            print(
                f"B [{row['label_b']}]: "
                f"{row['text_b']}"
            )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    cross_label_count = 0

    if not pairs.empty:
        cross_label_count = int(
            (
                pairs["label_a"]
                !=
                pairs["label_b"]
            ).sum()
        )

    summary = {
        "samples": len(df),
        "embedding_dimensions": int(
            embeddings.shape[1]
        ),
        "exact_duplicate_pairs": len(
            exact
        ),
        "similar_pairs": len(
            pairs
        ),
        "cross_label_similar_pairs": (
            cross_label_count
        ),
        "similarity_threshold": (
            args.threshold
        ),
        "embedding_model": args.model,
        "batch_size": args.batch_size,
        "checkpoint_interval": (
            args.checkpoint
        ),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Done.")
    print(
        f"Report directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()