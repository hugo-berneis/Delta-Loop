"""TDD tests for clustering/failure_clusterer.py — written BEFORE implementation per spec."""
import json
from collections import Counter

import numpy as np
import pytest

from deltaloop.clustering.failure_clusterer import sample_balanced
from deltaloop.storage.models import PreferencePair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pairs_with_labels(label_list: list[int]) -> list[PreferencePair]:
    pairs = []
    for i, label in enumerate(label_list):
        p = PreferencePair(
            task_id=f"task_{i:04d}",
            iteration=1,
            chosen_trace=json.dumps(["Correct step"]),
            rejected_trace=json.dumps(["Wrong step"]),
            failure_mode="HALLUCINATION",
            failure_explanation=f"Failure explanation for pair {i}",
            quality_score=0.8,
            cluster_label=label,
        )
        pairs.append(p)
    return pairs


# ---------------------------------------------------------------------------
# sample_balanced — TDD required per spec
# ---------------------------------------------------------------------------

def test_no_cluster_exceeds_30_percent():
    # 4 clusters: one dominant (50 items). With 4 clusters, math is feasible:
    # max_per_cluster = floor(60 * 0.30) = 18, phase-1 total = 18+18+15+10 = 61 >= 60
    label_list = [0] * 50 + [1] * 25 + [2] * 15 + [3] * 10
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=60, max_cluster_fraction=0.30)
    counts = Counter(p.cluster_label for p in sampled)

    for label, count in counts.items():
        fraction = count / len(sampled)
        assert fraction <= 0.30 + 1e-9, (
            f"Cluster {label} contributed {count}/{len(sampled)} = {fraction:.3f} > 30%"
        )


def test_sample_returns_requested_n():
    label_list = [0] * 20 + [1] * 20 + [2] * 20
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=30)
    assert len(sampled) == 30


def test_handles_fewer_pairs_than_n():
    label_list = [0] * 5 + [1] * 5
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=100)
    assert len(sampled) == 10  # can't sample more than exist


def test_dominant_cluster_is_capped():
    # Cluster 0 has 80% of data — without balancing it would dominate
    label_list = [0] * 80 + [1] * 10 + [2] * 10
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    # With 3 balanced clusters of 100 total, n=60:
    # max_per_cluster = 18; cluster 0 capped at 18 (vs 80 available)
    sampled = sample_balanced(pairs, labels, n=30, max_cluster_fraction=0.30)
    counts = Counter(p.cluster_label for p in sampled)
    # Cluster 0 should NOT contribute all 80% of sample
    cluster_0_fraction = counts.get(0, 0) / len(sampled)
    assert cluster_0_fraction <= 0.50  # strictly less than its natural 80%


def test_returns_all_when_n_equals_total():
    label_list = [0] * 10 + [1] * 10
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=20)
    assert len(sampled) == 20


def test_empty_pairs_returns_empty():
    sampled = sample_balanced([], np.array([]), n=10)
    assert sampled == []


def test_returns_list_of_preference_pairs():
    label_list = [0] * 5 + [1] * 5
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=6)
    assert all(isinstance(p, PreferencePair) for p in sampled)


def test_no_duplicate_pairs_in_sample():
    label_list = [0] * 10 + [1] * 10
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=15)
    # No pair should appear twice
    ids = [id(p) for p in sampled]
    assert len(ids) == len(set(ids))


def test_single_cluster_returns_n():
    label_list = [0] * 30
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=20)
    assert len(sampled) == 20


def test_n_zero_returns_empty():
    label_list = [0] * 10
    pairs = make_pairs_with_labels(label_list)
    labels = np.array(label_list)

    sampled = sample_balanced(pairs, labels, n=0)
    assert sampled == []
