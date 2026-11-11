"""Class-aware support/query sampling for binary assay episodes."""

import numpy as np


def episode_is_possible(dataset, n_support, task_id):
    """Check that disjoint support and query sets can each contain both classes."""
    counts = tuple(len(indices) for indices in dataset.index_list[task_id])
    return n_support >= 1 and min(counts) >= 2 and sum(counts) >= 2 * n_support + 2


def dataset_sampler(dataset, n_support, n_query, tgt_id, inductive=False, rng=None):
    """Sample support and query examples from the labeled molecules of one assay.

    Reserve one query example per class, then fill a short support class from
    the other class. A short training query can repeat molecules from its own
    pool. At evaluation the query is every labeled molecule outside support.
    """
    if n_query < 2 and not inductive:
        raise ValueError("A training query needs at least two molecules.")
    if not episode_is_possible(dataset, n_support, tgt_id):
        counts = tuple(len(indices) for indices in dataset.index_list[tgt_id])
        raise ValueError(
            f"Task {tgt_id} has class counts {counts} and cannot form "
            f"disjoint two-class support and query sets with {2 * n_support} support examples."
        )

    rng = np.random if rng is None else rng
    negatives, positives = dataset.index_list[tgt_id]
    negative_capacity, positive_capacity = len(negatives) - 1, len(positives) - 1
    n_negative = min(negative_capacity, max(n_support, 2 * n_support - positive_capacity))
    n_positive = 2 * n_support - n_negative
    support_ids = (
        rng.choice(negatives, n_negative, replace=False).tolist()
        + rng.choice(positives, n_positive, replace=False).tolist()
    )

    support_set = set(support_ids)
    negative_query = [idx for idx in negatives if idx not in support_set]
    positive_query = [idx for idx in positives if idx not in support_set]
    query_pool = negative_query + positive_query
    if inductive:
        return dataset[support_ids], dataset[query_pool]

    query_ids = [int(rng.choice(negative_query)), int(rng.choice(positive_query))]
    reserved = set(query_ids)
    remaining = [idx for idx in query_pool if idx not in reserved]
    unique_count = min(n_query - 2, len(remaining))
    if unique_count:
        query_ids.extend(rng.choice(remaining, unique_count, replace=False).tolist())
    if len(query_ids) < n_query:
        query_ids.extend(rng.choice(query_pool, n_query - len(query_ids), replace=True).tolist())
    return dataset[support_ids], dataset[query_ids]
