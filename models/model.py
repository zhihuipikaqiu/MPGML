"""MPGML molecular encoders, multimodal fusion, and episode prediction."""

import torch
import torch.nn as nn

from dataset.molecular_features import get_descriptor_dim, get_fingerprint_dim
from .base_encoder import GNN_Encoder
from .multimodal import DescriptorEncoder, FingerprintEncoder, MolecularFusion
from .relation import Context_Encoder


class MPGML(nn.Module):
    """Encode molecules and predict an episode using a task–molecule graph."""

    def __init__(self, task_num, train_task_num, args):
        super().__init__()
        self.args = args
        self.use_fingerprint = getattr(args, 'use_fingerprint', False)
        self.use_descriptor = getattr(args, 'use_descriptor', False)
        self.modality_dropout = getattr(args, 'modality_dropout', 0.0)
        self.fp_gate_reg = getattr(args, 'fp_gate_reg', 0.0)
        self.desc_gate_reg = getattr(args, 'desc_gate_reg', 0.0)
        self.fusion_residual = getattr(args, 'fusion_residual', True)

        self.mol_encoder = GNN_Encoder(num_layer=args.mol_num_layer,
                                       emb_dim=args.emb_dim,
                                       JK=args.JK,
                                       drop_ratio=args.mol_dropout,
                                       graph_pooling=args.mol_graph_pooling,
                                       gnn_type=args.mol_gnn_type,
                                       batch_norm=args.mol_batch_norm,
                                       load_path=args.mol_pretrain_load_path)

        n_modalities = 1
        if self.use_fingerprint:
            fp_dim = get_fingerprint_dim(getattr(args, 'fp_type', 'mixed'))
            self.fp_encoder = FingerprintEncoder(
                fp_dim=fp_dim,
                hidden_dim=args.emb_dim,
                fp_type=getattr(args, 'fp_type', 'mixed'),
                encoder_type=getattr(args, 'fp_encoder_type', 'typewise'),
                intermediate_dim=getattr(args, 'fp_intermediate_dim', 256),
                dropout=getattr(args, 'fusion_dropout', args.mol_dropout),
                use_gate=getattr(args, 'use_fp_gate', True),
            )
            n_modalities += 1
        else:
            self.fp_encoder = None

        if self.use_descriptor:
            self.desc_encoder = DescriptorEncoder(
                desc_dim=get_descriptor_dim(),
                hidden_dim=args.emb_dim,
                intermediate_dim=getattr(args, 'desc_intermediate_dim', 128),
                dropout=getattr(args, 'fusion_dropout', args.mol_dropout),
                use_gate=getattr(args, 'use_desc_gate', True),
            )
            n_modalities += 1
        else:
            self.desc_encoder = None

        self.fusion = MolecularFusion(
            hidden_dim=args.emb_dim,
            n_modalities=n_modalities,
            method=getattr(args, 'fusion_method', 'transformer'),
            num_layers=getattr(args, 'fusion_layers', 2),
            num_heads=getattr(args, 'fusion_heads', 4),
            dropout=getattr(args, 'fusion_dropout', args.mol_dropout),
            use_cross_modal_attn=getattr(args, 'use_cross_modal_attn', False),
            use_l2_norm=getattr(args, 'use_l2_norm', False),
            temperature_init=getattr(args, 'fusion_temperature', 1.0),
        )
        if n_modalities > 1 and self.fusion_residual:
            self.fusion_residual_gate = nn.Parameter(
                torch.tensor(float(getattr(args, 'fusion_residual_gate_init', -3.0)))
            )
            self.fusion_delta = nn.Sequential(
                nn.LayerNorm(args.emb_dim),
                nn.Linear(args.emb_dim, args.emb_dim),
                nn.GELU(),
                nn.Linear(args.emb_dim, args.emb_dim),
            )
            nn.init.zeros_(self.fusion_delta[-1].weight)
            nn.init.zeros_(self.fusion_delta[-1].bias)
        else:
            self.fusion_residual_gate = None
            self.fusion_delta = None

        self.relation_net = Context_Encoder(in_dim=args.emb_dim,
                                            num_layer=args.rel_layer,
                                            edge_n_layer=args.rel_edge_n_layer,
                                            edge_hidden_dim=args.rel_edge_hidden_dim,
                                            total_tasks=task_num,
                                            train_tasks=train_task_num,
                                            batch_norm=args.rel_batch_norm,
                                            top_k=args.rel_top_k,
                                            dropout=args.rel_dropout,
                                            pre_dropout=args.rel_pre_dropout,
                                            nan_w=args.rel_nan_w,
                                            nan_type=args.rel_nan_type,
                                            edge_type=args.rel_edge_type)

    def _encode_modality_items(self, data):
        graph_feat = self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        features = [('graph', graph_feat)]

        if self.use_fingerprint:
            fp = self._get_data_feature(data, 'fp')
            fp_feat = self.fp_encoder(fp)
            features.append(('fingerprint', fp_feat))

        if self.use_descriptor:
            desc = self._get_data_feature(data, 'desc')
            desc_feat = self.desc_encoder(desc)
            features.append(('descriptor', desc_feat))

        return features

    def _apply_modality_ablation(self, features, ablate_modalities=None):
        if not ablate_modalities:
            return features

        ablate_modalities = {str(name).lower() for name in ablate_modalities}
        return [
            (name, torch.zeros_like(feature) if name in ablate_modalities else feature)
            for name, feature in features
        ]

    def _fuse_encoded_features(self, features):
        fused = self.fusion(features)
        if self.fusion_delta is not None:
            delta = self.fusion_delta(fused)
            gate = torch.sigmoid(self.fusion_residual_gate)
            return features[0] + gate * delta
        return fused

    def available_modalities(self):
        """Return modality names in the order expected by the fusion module."""
        modalities = ['graph']
        if self.use_fingerprint:
            modalities.append('fingerprint')
        if self.use_descriptor:
            modalities.append('descriptor')
        return modalities

    def encode_mol(self, data, ablate_modalities=None):
        """Return molecular features with optional ablation and training dropout."""
        named_features = self._encode_modality_items(data)
        named_features = self._apply_modality_ablation(named_features, ablate_modalities)
        features = [feature for _, feature in named_features]
        features = self._apply_modality_dropout(features)
        return self._fuse_encoded_features(features)

    def extract_modal_features(self, data, ablate_modalities=None):
        """Return individual and fused features without modality dropout."""
        named_features = self._encode_modality_items(data)
        named_features = self._apply_modality_ablation(named_features, ablate_modalities)
        features = [feature for _, feature in named_features]
        output = {name: feature for name, feature in named_features}
        output['fused'] = self._fuse_encoded_features(features)
        return output

    def _get_data_feature(self, data, name):
        value = getattr(data, name, None)
        if value is None:
            raise ValueError(
                f"Missing Data.{name}. Create FewshotMolDataset with the matching "
                "use_fingerprint/use_descriptor option enabled."
            )
        if value.dim() == 1:
            value = value.view(1, -1)
        return value

    def _apply_modality_dropout(self, features):
        if (not self.training) or self.modality_dropout <= 0 or len(features) <= 1:
            return features

        kept = [features[0]]
        for feature in features[1:]:
            if torch.rand(1, device=feature.device).item() < self.modality_dropout:
                kept.append(torch.zeros_like(feature))
            else:
                kept.append(feature)
        return kept

    def get_regularization_loss(self):
        """Penalize active fingerprint and descriptor gates when configured."""
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.fp_encoder is not None and self.fp_gate_reg > 0:
            reg_loss = reg_loss + self.fp_gate_reg * self.fp_encoder.get_regularization_loss()
        if self.desc_encoder is not None and self.desc_gate_reg > 0:
            reg_loss = reg_loss + self.desc_gate_reg * self.desc_encoder.get_regularization_loss()
        return reg_loss

    def forward(self, s_data, q_data, s_y, q_y, sampled_task, bottleneck=False, test=False, relation=False,
                s_ablate_modalities=None, q_ablate_modalities=None):
        """Predict support and query labels for target-first task columns.

        Returns prediction tuples, optional relation predictions, an episode
        embedding, the bottleneck penalty, and the auxiliary-task retention rate.
        """
        s_feat = self.encode_mol(s_data, ablate_modalities=s_ablate_modalities)
        q_feat = self.encode_mol(q_data, ablate_modalities=q_ablate_modalities)
        sample_feat = torch.cat([s_feat, q_feat], dim=0)  # [n_support + n_query, emb_dim]
        pred, rel_pred, graph_f, KL_Loss, preserve_rate = self.relation_net.forward_inductive(sample_feat, sampled_task,
                                                                                              s_y, q_y,
                                                                                              bottleneck=bottleneck,
                                                                                              test=test, relation=relation)
        return pred, rel_pred, graph_f, KL_Loss, preserve_rate
