"""Class-aware support/query sampling for binary assay episodes."""

import numpy as np


def _sample_without_replacement(candidates, size):
    return np.random.choice(candidates, size, replace=False).tolist()


def dataset_sampler(dataset, n_support, n_query, tgt_id, inductive=False):
    """Sample ``2 * n_support`` support molecules for one target assay.

    Support is balanced when both classes have enough molecules. Otherwise,
    reserve one query example in the smaller class and fill support from the
    other class. Training queries contain both classes and total ``n_query``
    molecules; inductive evaluation uses every labeled non-support molecule.
    Missing target labels never enter either set.
    """
    tgt_index_list = dataset.index_list[tgt_id]
    class0_num, class1_num = len(tgt_index_list[0]), len(tgt_index_list[1])
    if n_support < 1:
        raise ValueError('n_support must be at least 1 per class.')
    if min(class0_num, class1_num) < 1 or class0_num + class1_num < 2 * n_support + 2:
        raise ValueError(
            f'Task {tgt_id} has class counts ({class0_num}, {class1_num}); '
            f'cannot draw {2 * n_support} support molecules and reserve both query classes.'
        )
    if not inductive and (n_query < 2 or n_query > class0_num + class1_num - 2 * n_support):
        raise ValueError(
            f'Task {tgt_id} requires 2 <= n_query <= '
            f'{class0_num + class1_num - 2 * n_support}; got {n_query}.'
        )

    if class0_num > n_support and class1_num > n_support:
        support_list_i_0 = _sample_without_replacement(tgt_index_list[0], n_support)
        support_list_i_1 = _sample_without_replacement(tgt_index_list[1], n_support)
    elif class0_num <= n_support < class1_num:
        support_list_i_0 = _sample_without_replacement(tgt_index_list[0], class0_num - 1)
        support_list_i_1 = _sample_without_replacement(tgt_index_list[1], 2 * n_support - class0_num + 1)
    else:
        support_list_i_0 = _sample_without_replacement(tgt_index_list[0], 2 * n_support - class1_num + 1)
        support_list_i_1 = _sample_without_replacement(tgt_index_list[1], class1_num - 1)
    support_list = support_list_i_0 + support_list_i_1

    if not inductive:
        query_candi_i_0 = [idx for idx in tgt_index_list[0] if idx not in support_list]
        query_candi_i_1 = [idx for idx in tgt_index_list[1] if idx not in support_list]
        query_list = _sample_without_replacement(query_candi_i_0, 1) + _sample_without_replacement(query_candi_i_1, 1)
        query_candi = [idx for idx in query_candi_i_0 + query_candi_i_1 if idx not in query_list]
        query_list += _sample_without_replacement(query_candi, n_query - 2)
    else:
        query_list = [idx for idx in tgt_index_list[0] + tgt_index_list[1]
                      if idx not in support_list]
    return dataset[support_list], dataset[query_list]
