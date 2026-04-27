import math
import random
from collections import Counter, defaultdict

import numpy as np
from loguru import logger

from deltaloop.storage.models import PreferencePair


# ---------------------------------------------------------------------------
# sample_balanced
# ---------------------------------------------------------------------------

def sample_balanced(
    pairs: list[PreferencePair],
    labels: np.ndarray,
    n: int,
    max_cluster_fraction: float = 0.30,
) -> list[PreferencePair]:
    """Sample n pairs with no cluster contributing more than max_cluster_fraction.

    Algorithm:
    1. Cap each cluster at floor(n * max_cluster_fraction).
    2. If total < n and more data exists, fill remaining slots from clusters
       with leftover capacity (smallest clusters first to stay balanced).
    3. Return min(n, available) items — never more than exist.
    """
    if not pairs or n == 0:
        return []

    n_target = min(n, len(pairs))
    max_per_cluster = max(1, math.floor(n_target * max_cluster_fraction))

    # Group indices by cluster label
    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_indices[int(label)].append(i)

    # Shuffle within each cluster for random selection
    for cid in cluster_indices:
        random.shuffle(cluster_indices[cid])

    # Phase 1: apply per-cluster cap
    selected: list[int] = []
    for cid in sorted(cluster_indices):
        take = min(len(cluster_indices[cid]), max_per_cluster)
        selected.extend(cluster_indices[cid][:take])

    # Phase 2: fill remaining slots by relaxing the cap
    # Prioritise smallest clusters (more underrepresented) first.
    if len(selected) < n_target:
        selected_set = set(selected)
        extra: list[int] = []
        for cid in sorted(cluster_indices, key=lambda c: len(cluster_indices[c])):
            for idx in cluster_indices[cid]:
                if idx not in selected_set:
                    extra.append(idx)

        needed = n_target - len(selected)
        selected.extend(extra[:needed])

    random.shuffle(selected)
    return [pairs[i] for i in selected[:n_target]]


# ---------------------------------------------------------------------------
# embed_explanations
# ---------------------------------------------------------------------------

def embed_explanations(pairs: list[PreferencePair]) -> np.ndarray:
    """Embed failure_explanation strings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [p.failure_explanation or "" for p in pairs]
    embeddings: np.ndarray = model.encode(texts, show_progress_bar=False)
    logger.debug(f"embed_explanations: encoded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# cluster_pairs
# ---------------------------------------------------------------------------

def cluster_pairs(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Run KMeans and return cluster label array."""
    from sklearn.cluster import KMeans  # type: ignore[import-untyped]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels: np.ndarray = kmeans.fit_predict(embeddings)
    logger.info(f"cluster_pairs: k={k} inertia={kmeans.inertia_:.2f}")
    return labels


# ---------------------------------------------------------------------------
# run_clustering (top-level)
# ---------------------------------------------------------------------------

async def run_clustering(repo, k: int) -> dict[int, int]:
    """Embed, cluster, and write labels back to DB.

    Returns cluster_label → count mapping.
    """
    from deltaloop.config import settings

    pairs = await repo.get_all_pairs()
    if not pairs:
        logger.warning("run_clustering: no pairs found, skipping")
        return {}

    logger.info(f"run_clustering: clustering {len(pairs)} pairs into k={k}")
    embeddings = embed_explanations(pairs)
    labels = cluster_pairs(embeddings, k)

    updates = [(pair.id, int(label)) for pair, label in zip(pairs, labels) if pair.id is not None]
    await repo.update_cluster_labels(updates)

    distribution = dict(Counter(int(l) for l in labels))
    logger.info(f"run_clustering: distribution={distribution}")
    return distribution
