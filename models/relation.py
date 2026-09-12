"""Task–molecule relation graph, relation objective, and task bottleneck."""

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from .graph_connector import InductiveGraphConnector


class EdgeUpdateNetwork(nn.Module):
    """Infer directed top-k molecule edges from pairwise feature differences."""

    def __init__(self, in_dim, hidden_dim, n_layer=3, top_k=5, batch_norm=False, dropout=.0):
        super(EdgeUpdateNetwork, self).__init__()
        self.top_k = top_k

        num_dims_list = [hidden_dim] * n_layer
        if n_layer > 1:
            num_dims_list[0] = 2 * hidden_dim
        if n_layer > 3:
            num_dims_list[1] = 2 * hidden_dim

        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=num_dims_list[l - 1] if l > 0 else in_dim,
                                                       out_channels=num_dims_list[l],
                                                       kernel_size=1,
                                                       bias=False)  # in: [N, C_in, H, W]
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_dims_list[l], )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=num_dims_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat):
        """Map ``[n_molecules, emb_dim]`` features to a square adjacency matrix."""
        x_i = node_feat.unsqueeze(1)
        x_j = node_feat.unsqueeze(0)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 0, 2)
        x_ij = torch.exp(-x_ij)
        x_ij = x_ij.unsqueeze(0)
        sim_val = self.sim_network(x_ij).squeeze()
        diag_mask = 1.0 - torch.eye(node_feat.size(0)).to(node_feat.device)
        adj_val = torch.sigmoid(sim_val) * diag_mask
        if self.top_k >= 0:
            k = min(self.top_k, adj_val.size(0))
            _, indices = torch.topk(adj_val, k)
            mask = torch.zeros_like(adj_val)
            mask = mask.scatter(1, indices, 1)
            adj_val = adj_val * mask
        return adj_val


class NodeUpdateNetwork(MessagePassing):
    """Aggregate weighted neighbor messages and optional edge-type embeddings."""

    def __init__(self, in_emb_dim, out_emb_dim, num_bond_type=4, norm=False, batch_norm=False, edge_type=True,
                 dropout=0., aggr='mean'):
        super(NodeUpdateNetwork, self).__init__(aggr=aggr)
        self.edge_type = edge_type
        self.neigh_linear = nn.Linear(in_emb_dim, out_emb_dim)
        self.root_linear = nn.Linear(in_emb_dim, out_emb_dim)
        self.edge_emb = nn.Embedding(num_bond_type, out_emb_dim)
        nn.init.xavier_uniform_(self.edge_emb.weight.data)
        self.norm = norm
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_emb_dim)
        else:
            self.batch_norm = None
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr, edge_weight):
        """
        :param x: [num_node, d]
        :param edge_index: [2, num_edge]
        :param edge_attr: [num_edge, num_attr]
        :param edge_weight: [num_edge, 1]
        :return:
        """
        edge_embeddings = self.edge_emb(edge_attr[:, 0])

        neigh_x = self.neigh_linear(x)
        msg = self.propagate(edge_index, x=neigh_x, edge_attr=edge_embeddings, edge_weight=edge_weight)
        msg = msg + self.root_linear(x)
        if self.norm:
            msg = F.normalize(msg, p=2, dim=-1)
        if self.batch_norm:
            msg = self.batch_norm(msg)
        return self.dropout(self.act(msg))

    def message(self, x_j, edge_attr, edge_weight):
        if self.edge_type:
            return (x_j + edge_attr) * edge_weight
        else:
            return x_j * edge_weight


class Context_Encoder(nn.Module):
    """Predict molecule–target pairs after message passing over episode nodes.

    Nodes are ordered as target task, auxiliary tasks, support molecules, and
    query molecules. Parameter names are retained for checkpoint compatibility.
    """

    def __init__(self,
                 in_dim,
                 num_layer,
                 edge_hidden_dim,
                 edge_n_layer,
                 total_tasks,
                 train_tasks,
                 batch_norm=False,
                 edge_type=True,
                 top_k=-1,
                 dropout=0.,
                 pre_dropout=0.,
                 nan_w=0.,
                 nan_type='nan'):
        super(Context_Encoder, self).__init__()
        self.dropout = dropout
        self.total_tasks = total_tasks
        self.num_layer = num_layer
        self.nan_w = nan_w
        self.nan_type = nan_type
        if pre_dropout > 0:
            self.pre_dropout = nn.Dropout(pre_dropout)
        else:
            self.pre_dropout = None
        self.task_emb = nn.Embedding(total_tasks, in_dim)
        self.task_emb.weight.data[train_tasks:, :] = 0
        for i in range(num_layer):
            module_edge = EdgeUpdateNetwork(in_dim=in_dim, hidden_dim=edge_hidden_dim, n_layer=edge_n_layer,
                                            top_k=top_k, batch_norm=batch_norm, dropout=dropout)
            module_node = NodeUpdateNetwork(in_emb_dim=in_dim, out_emb_dim=in_dim,
                                            dropout=dropout, batch_norm=batch_norm, edge_type=edge_type)
            self.add_module(f'node_layer{i}', module_node)
            self.add_module(f'edge_layer{i}', module_edge)
        # Retained for existing state dictionaries and initialization order.
        # Episode pooling below does not call this module.
        self.graph_pooling = nn.Sequential(nn.Linear(in_dim, in_dim//2), nn.ReLU(),
                                           nn.Linear(in_dim//2, in_dim))
        self.classifier = nn.Sequential(nn.Linear(2 * in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(),
                                        nn.Linear(in_dim, 1))
        self.rel_classifier = nn.Sequential(nn.Linear(2 * in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(),
                                            nn.Linear(in_dim, 1))
        self.compressor = nn.Sequential(nn.Linear(2 * in_dim, in_dim),nn.BatchNorm1d(in_dim),nn.ReLU(),
                                        nn.Linear(in_dim, 1))

        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def compress(self, task_features):
        """Sample relaxed Bernoulli gates for the auxiliary-task bottleneck."""
        p = self.compressor(task_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(task_features.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p

    def forward_inductive(self, sample_emb, task_id, support_y, query_y, bottleneck=False, test=False, relation=False):
        """Build one graph for the support set and the current query batch."""
        task_emb = self.task_emb(task_id)  # [n_task, d]
        n_task = support_y.shape[1]
        s_tri, s_rel_labs = self.triple_sample(support_y, 0, 0)
        q_tri, q_rel_labs = self.triple_sample(query_y, 0, 1)
        if self.pre_dropout:
            sample_emb = self.pre_dropout(sample_emb)  # [n_support + n_query, d]
            task_emb = self.pre_dropout(task_emb)
        tgt_s_y, tgt_q_y, tgt_s_idx, tgt_q_idx, edge_ls, edge_type_ls, edge_w_ls = \
            InductiveGraphConnector.connect_task_and_sample(support_y, query_y, self.nan_w, self.nan_type)
        input_emb = torch.cat([task_emb, sample_emb], dim=0)  # [n_task + n_support + n_query, d]

        if (not test) & bottleneck:
            mask = torch.zeros(input_emb.shape[0], dtype=torch.bool).to(input_emb.device)
            mask[len(task_emb):] = True
            mask[0] = True
            com_graph_emb = (input_emb[0] + torch.sigmoid(input_emb[len(task_id):].mean(0))).repeat(input_emb.shape[0], 1)
            com_input = torch.concat((input_emb, com_graph_emb), dim=-1)[~mask]
            lambda_pos, p = self.compress(com_input)
            lambda_pos = lambda_pos.reshape(-1, 1)
            lambda_neg = 1 - lambda_pos
            preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()
            static_task_emb = input_emb[~mask].clone().detach()

            task_emb_mean = static_task_emb.mean(dim=0)
            task_emb_std = static_task_emb.std(dim=0) + 0.1
            noisy_task_emb_mean = lambda_pos * input_emb[~mask] + lambda_neg * task_emb_mean
            noisy_task_emb_std = lambda_neg * task_emb_std
            noisy_emb_feature = noisy_task_emb_mean + torch.randn_like(noisy_task_emb_mean) * noisy_task_emb_std
            epsilon = 1e-7
            KL_Loss = 0.5 * ((noisy_task_emb_std ** 2) / (task_emb_std + epsilon) ** 2).mean(dim=-1).sum() + \
                      (((noisy_task_emb_mean - task_emb_mean) / (task_emb_std + epsilon)) ** 2).mean(dim=-1).sum()
            input_emb = input_emb.masked_scatter(
                (~mask).unsqueeze(-1).expand_as(input_emb),
                noisy_emb_feature.reshape(-1)
            )
        else:
            KL_Loss = 0
            preserve_rate = 0

        for i in range(self.num_layer):
            sample_adj = self._modules['edge_layer{}'.format(i)](sample_emb)
            edges, edge_types, edge_ws = InductiveGraphConnector.connect_graph(sample_adj, edge_ls, edge_type_ls,
                                                                               edge_w_ls, n_task)
            data = Data(x=input_emb, edge_index=edges, edge_type=edge_types, edge_w=edge_ws)
            input_emb = self._modules['node_layer{}'.format(i)](data.x, data.edge_index, data.edge_type, data.edge_w)
            input_emb = input_emb.contiguous()
            task_emb, sample_emb = input_emb[:len(task_id), :], input_emb[len(task_id):, :]
        # Pair each molecule node with the target-task node at index zero.
        support_sample = torch.cat([input_emb[tgt_s_idx[:, 0], :],
                                    input_emb[tgt_s_idx[:, 1], :]], dim=-1)  # [n_support, 2*d]
        query_sample = torch.cat([input_emb[tgt_q_idx[:, 0], :],
                                  input_emb[tgt_q_idx[:, 1], :]], dim=-1)  # [n_query, 2*d]
        support_logit = self.classifier(support_sample)

        query_logit = self.classifier(query_sample)  # [n_q, 1]

        # relation learning
        if relation:
            rel_tri = torch.cat([t for t in [s_tri, q_tri] if t is not None], dim=0) if (
                        (s_tri is not None) or (q_tri is not None)) else None
            rel_labels = torch.cat([t for t in [s_rel_labs, q_rel_labs] if t is not None], dim=-1) if (
                        (s_tri is not None) or (q_tri is not None)) else None
            if rel_tri is not None and rel_tri.size(0) == 1:
                rel_tri = rel_tri.repeat(2, 1)
                rel_labels = rel_labels.repeat(2)
            rel_logit = self.rel_classifier(
                torch.mul(torch.cat([input_emb[rel_tri[:, 0]], input_emb[rel_tri[:, 1]]], dim=-1),
                          torch.cat([input_emb[rel_tri[:, 0]], input_emb[rel_tri[:, 2]]], dim=-1))).squeeze(
                -1) if rel_tri is not None else None
        else:
            rel_logit = None
            rel_labels = None

        graph_emb = input_emb[0] + torch.sigmoid(input_emb[1:].mean(0))
        pred = (support_logit, query_logit, tgt_s_y, tgt_q_y)
        rel_pred = (rel_logit, rel_labels)
        return pred, rel_pred, graph_emb, KL_Loss, preserve_rate

    def triple_sample(self, vector_y, idx, pos=0):
        """Enumerate task-label pairs for each molecule and their disagreement."""
        device = vector_y.device
        triples_list = []
        labels_list = []
        for i, temp in enumerate(vector_y):
            temp_idx = i + idx
            valid_idx = torch.where(~torch.isnan(temp))[0]
            if valid_idx.numel() < 2:
                continue
            valid_idx = valid_idx[pos:]
            grid_i, grid_j = torch.meshgrid(valid_idx, valid_idx, indexing="ij")
            mask = grid_i < grid_j
            comb = torch.stack([grid_i[mask], grid_j[mask]], dim=1)
            diffs = (temp[comb[:, 0]] - temp[comb[:, 1]]).pow(2)
            mol_idx = torch.full((comb.size(0), 1), temp_idx, dtype=torch.long, device=device)
            triples_list.append(torch.cat([mol_idx, comb.to(device)], dim=1))
            labels_list.append(diffs.to(device))
        if len(triples_list) > 0:
            triples = torch.cat(triples_list, dim=0)  # (M, 3), long
            labels = torch.cat(labels_list, dim=0).to(vector_y.dtype)
        else:
            triples, labels = None, None
        return triples, labels
