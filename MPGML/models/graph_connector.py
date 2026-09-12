"""Construct typed task–molecule and learned molecule–molecule graph edges."""

import torch

nan_type_dict = {'0': 1, '1': 2, 'nan': 3}


def get_edge(label_matrix, label, shift, bi_direct=True):
    """Return molecule/task index pairs matching a label or missing values."""
    if label is None:
        edge = torch.nonzero(torch.isnan(label_matrix), as_tuple=False)
    else:
        edge = torch.nonzero(label_matrix == label, as_tuple=False)
    edge[:, 0] = edge[:, 0] + shift
    if bi_direct:
        edge_copy = edge.clone()
        edge_copy[:, 0], edge_copy[:, 1] = edge[:, 1], edge[:, 0]
        edge = torch.cat([edge, edge_copy])  # [n_edge, 2]
    return edge


class InductiveGraphConnector:
    """Build one graph using target-first label columns for an episode."""

    @staticmethod
    def connect_task_and_sample(support_y, query_y, nan_w=0., nan_type='nan'):
        """Return target labels, molecule/target index pairs, and typed edges.

        ``support_y`` and ``query_y`` have shape ``[n_molecules, n_tasks]``.
        Edge types are 1 for negative, 2 for positive, and the configured type
        for missing labels. Query target labels are excluded from edge input.
        See the implementation audit for the retained query-task index issue.
        """
        nan_idx = nan_type_dict[nan_type]
        tgt_s_y, tgt_q_y = support_y[:, [0]], query_y[:, [0]]  # [n_s,1], [n_q,1]
        n_task = support_y.shape[1]
        n_q, n_s = query_y.shape[0], support_y.shape[0]
        auxi_query_y = query_y[:, 1:]
        support_edge_0 = get_edge(support_y, 0, n_task)
        support_edge_1 = get_edge(support_y, 1, n_task)
        query_edge_0 = get_edge(auxi_query_y, 0, n_task + n_s)
        query_edge_1 = get_edge(auxi_query_y, 1, n_task + n_s)
        if nan_w == 0:
            edge = torch.cat([support_edge_0, query_edge_0,
                              support_edge_1, query_edge_1])
            edge_type = [1] * (len(support_edge_0) + len(query_edge_0)) + \
                        [2] * (len(support_edge_1) + len(query_edge_1))
            edge_type = torch.tensor(edge_type).to(edge.device)
            edge_w = torch.tensor([1.] * len(edge_type)).to(edge.device)
        else:
            support_edge_nan = get_edge(support_y, None, n_task)
            query_edge_nan = get_edge(auxi_query_y, None, n_task + n_s)
            edge = torch.cat([support_edge_0, query_edge_0,
                              support_edge_1, query_edge_1,
                              support_edge_nan, query_edge_nan])
            edge_0_n, edge_1_n, edge_nan_n = len(support_edge_0) + len(query_edge_0), \
                                             len(support_edge_1) + len(query_edge_1), \
                                             len(support_edge_nan) + len(query_edge_nan)
            edge_type = [1] * edge_0_n + [2] * edge_1_n + [nan_idx] * edge_nan_n
            edge_type = torch.tensor(edge_type).to(edge.device)
            edge_w = torch.tensor([1.] * (edge_0_n + edge_1_n) + [nan_w] * edge_nan_n).to(edge.device)
        edge = edge.transpose(0, 1)
        edge_w = edge_w.unsqueeze(-1)
        edge_type = edge_type.unsqueeze(-1)

        tgt_s_idx = torch.tensor([list(range(n_task, n_task + n_s)), [0] * n_s]).transpose(0, 1)
        tgt_q_idx = torch.tensor([list(range(n_task + n_s, n_task + n_s + n_q)), [0] * n_q]).transpose(0, 1)
        tgt_s_idx, tgt_q_idx = tgt_s_idx.to(support_y.device), tgt_q_idx.to(support_y.device)
        return tgt_s_y, tgt_q_y, tgt_s_idx, tgt_q_idx, edge, edge_type, edge_w

    @staticmethod
    def connect_graph(adj, edge, edge_type, edge_w, n_task):
        """Append nonzero square-adjacency edges with molecule edge type zero.

        ``adj`` has shape ``[n_support + n_query, n_support + n_query]``.
        Task nodes precede molecule nodes, so molecule indices shift by
        ``n_task``. Return edge indices ``[2, E]``, types, and weights ``[E, 1]``.
        """
        sample_edge = torch.nonzero(adj > 0, as_tuple=False)
        sample_w = adj[sample_edge[:, 0], sample_edge[:, 1]]
        if sample_edge.shape[0] != 0:
            sample_edge = sample_edge + n_task
            edge = torch.cat([edge, sample_edge.transpose(0, 1)], dim=1)
            edge_type = torch.cat(
                [edge_type, torch.tensor([0] * sample_edge.shape[0]).unsqueeze(1).to(adj.device)], dim=0)
            edge_w = torch.cat([edge_w, sample_w.unsqueeze(1)], dim=0)

        return edge, edge_type, edge_w
