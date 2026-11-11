"""Shared MPGML encoder and support-set prototype initialization."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.molecular_features import get_descriptor_dim, get_fingerprint_dim

from .base_encoder import GNN_Encoder
from .multimodal import DescriptorEncoder, FingerprintEncoder, MolecularFusion


class MPGML(nn.Module):
    """Encode a molecule and adapt a two-class head on the support set."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mol_encoder = GNN_Encoder(
            num_layer=args.mol_num_layer,
            emb_dim=args.emb_dim,
            JK=args.JK,
            drop_ratio=args.mol_dropout,
            graph_pooling=args.mol_graph_pooling,
            gnn_type=args.mol_gnn_type,
            batch_norm=args.mol_batch_norm,
            load_path=args.mol_pretrain_load_path,
        )
        self.fp_encoder = None
        if args.use_fingerprint:
            self.fp_encoder = FingerprintEncoder(
                fp_dim=get_fingerprint_dim(args.fp_type),
                hidden_dim=args.emb_dim,
                fp_type=args.fp_type,
                encoder_type=args.fp_encoder_type,
                intermediate_dim=args.fp_intermediate_dim,
                dropout=args.fusion_dropout,
                use_gate=args.use_fp_gate,
            )

        self.desc_encoder = None
        if args.use_descriptor:
            self.desc_encoder = DescriptorEncoder(
                desc_dim=get_descriptor_dim(),
                hidden_dim=args.emb_dim,
                intermediate_dim=args.desc_intermediate_dim,
                dropout=args.fusion_dropout,
                use_gate=args.use_desc_gate,
            )

        self.fusion = MolecularFusion(
            hidden_dim=args.emb_dim,
            use_fingerprint=args.use_fingerprint,
            method=args.fusion_method,
            num_layers=args.fusion_layers,
            num_heads=args.fusion_heads,
            dropout=args.fusion_dropout,
            use_cross_modal_attn=args.use_cross_modal_attn,
            fusion_residual=args.fusion_residual,
            residual_gate_init=args.fusion_residual_gate_init,
        )
        if args.use_l2_norm:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(args.fusion_temperature)))
        else:
            self.register_parameter("log_temperature", None)

        self.shared_head = None
        if not args.prototype_guidance:
            self.shared_head = nn.Linear(args.emb_dim, 2)

    def available_modalities(self):
        modalities = ["graph"]
        if self.fp_encoder is not None:
            modalities.append("fingerprint")
        if self.desc_encoder is not None:
            modalities.append("descriptor")
        return modalities

    @staticmethod
    def _feature(data, name):
        value = getattr(data, name, None)
        if value is None:
            raise ValueError(f"Missing Data.{name}. Load the dataset with that modality enabled.")
        return value.unsqueeze(0) if value.ndim == 1 else value

    def _encode_modalities(self, data, ablate_modalities, modality_dropout):
        masked = {name.lower() for name in ablate_modalities or ()}
        if modality_dropout and self.training and self.args.modality_dropout > 0:
            for name in self.available_modalities()[1:]:
                if torch.rand(()).item() < self.args.modality_dropout:
                    masked.add(name)

        graph = self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        if "graph" in masked:
            graph = torch.zeros_like(graph)
        features = {"graph": graph}

        fingerprint = None
        if self.fp_encoder is not None:
            fingerprint = self.fp_encoder(self._feature(data, "fp"))
            if "fingerprint" in masked:
                fingerprint = torch.zeros_like(fingerprint)
            features["fingerprint"] = fingerprint

        fused = self.fusion(graph, fingerprint)
        if self.desc_encoder is not None:
            condition = self.desc_encoder(self._feature(data, "desc"))
            if "descriptor" in masked:
                condition = torch.zeros_like(condition)
            else:
                fused = self.desc_encoder.modulate(fused, condition)
            features["descriptor"] = condition

        if self.log_temperature is not None:
            norm = torch.linalg.vector_norm(fused, dim=-1, keepdim=True)
            fused = self.log_temperature.exp() * (fused / (norm + 1e-8))
        features["fused"] = fused
        return features

    def encode_mol(self, data, ablate_modalities=None):
        return self._encode_modalities(data, ablate_modalities, modality_dropout=True)["fused"]

    def extract_modal_features(self, data, ablate_modalities=None):
        return self._encode_modalities(data, ablate_modalities, modality_dropout=False)

    def forward(self, data, ablate_modalities=None):
        return self.encode_mol(data, ablate_modalities)

    def get_regularization_loss(self):
        penalty = next(self.parameters()).new_zeros(())
        if self.fp_encoder is not None and self.fp_encoder.gate is not None:
            penalty = penalty + self.args.fp_gate_reg * self.fp_encoder.gate.l1_loss()
        if self.desc_encoder is not None and self.desc_encoder.gate is not None:
            penalty = penalty + self.args.desc_gate_reg * self.desc_encoder.gate.l1_loss()
        return penalty

    def initialize_head(self, support, labels):
        """Use negative squared Euclidean distances as initial class logits."""
        if self.shared_head is not None:
            return self.shared_head.weight, self.shared_head.bias

        membership = F.one_hot(labels, num_classes=2).to(support.dtype)
        counts = membership.sum(dim=0)
        if torch.any(counts == 0):
            raise ValueError("Prototype initialization needs both classes in the support set.")
        prototypes = (membership.T @ support) / counts.unsqueeze(-1)
        return 2 * prototypes, -prototypes.square().sum(dim=-1)

    def adapt_head(self, support, labels, meta_training=False):
        """Differentiate support loss with respect to the episode's head only."""
        with torch.enable_grad():
            weight, bias = self.initialize_head(support, labels)
            if not meta_training:
                support = support.detach()
                weight = weight.detach().requires_grad_()
                bias = bias.detach().requires_grad_()

            for _ in range(self.args.inner_update_step):
                loss = F.cross_entropy(F.linear(support, weight, bias), labels)
                weight_grad, bias_grad = torch.autograd.grad(
                    loss, (weight, bias), create_graph=meta_training and bool(self.args.second_order),
                )
                weight = weight - self.args.inner_lr * weight_grad
                bias = bias - self.args.inner_lr * bias_grad

        if meta_training:
            return weight, bias
        return weight.detach(), bias.detach()
