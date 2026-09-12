"""Feature encoders and fusion operators for molecular modalities."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGate(nn.Module):
    """Learn a sigmoid weight for each input feature."""

    def __init__(self, dim, init_logit=0.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((dim,), float(init_logit)))

    def forward(self, x):
        return x * torch.sigmoid(self.logits).view(1, -1)

    def l1_loss(self):
        return torch.sigmoid(self.logits).mean()


def _make_mlp(in_dim, hidden_dim, intermediate_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, intermediate_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(intermediate_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
    )


class FingerprintEncoder(nn.Module):
    """Project fingerprints, optionally encoding the three families separately.

    The mixed fingerprint order is MACCS (167), ErG (441), and Morgan (881),
    with the last family stored under the retained ``pubchem`` option name.
    Typewise encoding learns a molecule-dependent softmax over these families.
    """

    def __init__(
        self,
        fp_dim,
        hidden_dim,
        fp_type="mixed",
        encoder_type="typewise",
        intermediate_dim=256,
        dropout=0.1,
        use_gate=True,
    ):
        super().__init__()
        self.fp_dim = fp_dim
        self.fp_type = fp_type.lower()
        self.encoder_type = encoder_type.lower()
        self.gate = FeatureGate(fp_dim) if use_gate else None

        self.typewise = self.fp_type == "mixed" and self.encoder_type == "typewise"
        if self.typewise:
            dims = [167, 441, 881]
            self.split_dims = dims
            self.part_encoders = nn.ModuleList(
                [_make_mlp(dim, hidden_dim, intermediate_dim, dropout) for dim in dims]
            )
            self.type_gate = nn.Linear(hidden_dim * len(dims), len(dims))
            self.out_norm = nn.LayerNorm(hidden_dim)
        else:
            self.encoder = _make_mlp(fp_dim, hidden_dim, intermediate_dim, dropout)

    def forward(self, fp):
        fp = torch.nan_to_num(fp.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if fp.dim() == 1:
            fp = fp.view(1, -1)
        if self.gate is not None:
            fp = self.gate(fp)

        if not self.typewise:
            return self.encoder(fp)

        parts = torch.split(fp, self.split_dims, dim=1)
        encoded_parts = [encoder(part) for encoder, part in zip(self.part_encoders, parts)]
        cat = torch.cat(encoded_parts, dim=1)
        weights = torch.softmax(self.type_gate(cat), dim=1)

        fused = torch.zeros_like(encoded_parts[0])
        for i, part in enumerate(encoded_parts):
            fused = fused + weights[:, i:i + 1] * part
        return self.out_norm(fused)

    def get_regularization_loss(self):
        if self.gate is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gate.l1_loss()


class DescriptorEncoder(nn.Module):
    """Normalize, gate, and project the molecular descriptor vector."""

    def __init__(
        self,
        desc_dim,
        hidden_dim,
        intermediate_dim=128,
        dropout=0.1,
        use_gate=True,
    ):
        super().__init__()
        self.desc_dim = desc_dim
        self.input_norm = nn.LayerNorm(desc_dim)
        self.gate = FeatureGate(desc_dim) if use_gate else None
        self.encoder = _make_mlp(desc_dim, hidden_dim, intermediate_dim, dropout)

    def forward(self, desc):
        desc = torch.nan_to_num(desc.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if desc.dim() == 1:
            desc = desc.view(1, -1)
        desc = self.input_norm(desc)
        if self.gate is not None:
            desc = self.gate(desc)
        return self.encoder(desc)

    def get_regularization_loss(self):
        if self.gate is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gate.l1_loss()


class ModalTransformerFusion(nn.Module):
    """Fuse modality tokens using learned positions and a pooled CLS token."""

    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, max_modalities=3):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.modality_embed = nn.Parameter(torch.zeros(1, max_modalities + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)

    def forward(self, tokens):
        cls = self.cls_token.expand(tokens.size(0), 1, tokens.size(2))
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.modality_embed[:, :x.size(1), :]
        x = self.encoder(x)
        return self.out_norm(x[:, 0])


class MolecularFusion(nn.Module):
    """Combine modality vectors, with optional token attention and output scaling.

    The historical ``sum`` option uses the mean of modality tokens followed by
    layer normalization. L2 scaling is applied before the model's optional
    graph residual, so the final residual representation need not have unit norm.
    """

    def __init__(
        self,
        hidden_dim,
        n_modalities,
        method="transformer",
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_cross_modal_attn=False,
        use_l2_norm=False,
        temperature_init=1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.method = method.lower()
        self.use_l2_norm = use_l2_norm

        if n_modalities < 1:
            raise ValueError("At least one modality is required.")

        if use_cross_modal_attn and n_modalities > 1:
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)
        else:
            self.cross_attn = None

        if n_modalities == 1:
            self.fusion = None
        elif self.method == "transformer":
            self.fusion = ModalTransformerFusion(
                hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_modalities=n_modalities,
            )
        elif self.method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * n_modalities, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_modalities),
            )
            self.out_norm = nn.LayerNorm(hidden_dim)
        elif self.method == "concat":
            self.concat = nn.Sequential(
                nn.Linear(hidden_dim * n_modalities, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        elif self.method == "sum":
            self.out_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        if use_l2_norm:
            self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        else:
            self.register_parameter("temperature", None)

    def forward(self, features):
        if len(features) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modalities, got {len(features)}.")

        if len(features) == 1:
            fused = features[0]
        else:
            tokens = torch.stack(features, dim=1)
            if self.cross_attn is not None:
                attn_out, _ = self.cross_attn(tokens, tokens, tokens)
                tokens = self.cross_norm(tokens + attn_out)

            if self.method == "transformer":
                fused = self.fusion(tokens)
            elif self.method == "attention":
                flat = tokens.reshape(tokens.size(0), -1)
                weights = torch.softmax(self.attention(flat), dim=1)
                fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
                fused = self.out_norm(fused)
            elif self.method == "concat":
                fused = self.concat(tokens.reshape(tokens.size(0), -1))
            elif self.method == "sum":
                fused = self.out_norm(tokens.mean(dim=1))
            else:
                raise ValueError(f"Unknown fusion method: {self.method}")

        if self.use_l2_norm:
            fused = F.normalize(fused, p=2, dim=-1) * math.sqrt(self.hidden_dim) * self.temperature
        return fused
