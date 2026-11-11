"""Fingerprint reconstruction and graph–fingerprint fusion for MPGML."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.molecular_features import MIXED_FINGERPRINT_TYPES, get_fingerprint_dim


# Table S1. The values give the number of buckets at each scale.
BUCKET_COUNTS = {
    "maccs": (16, 32, 64),
    "erg": (32, 64, 128),
    "pubchem": (64, 256, 512),
    "morgan": (64, 256, 512),
}


class FeatureGate(nn.Module):
    """Apply a learned sigmoid weight to each input feature."""

    def __init__(self, dim):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(dim))

    def forward(self, features):
        return features * torch.sigmoid(self.logits)

    def l1_loss(self):
        return torch.sigmoid(self.logits).sum()


def _make_mlp(in_dim, hidden_dim, intermediate_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, intermediate_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(intermediate_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
    )


class FingerprintScales(nn.Module):
    """Pool one permuted fingerprint at multiple resolutions."""

    def __init__(self, width, counts, hidden_dim, intermediate_dim, dropout, generator):
        super().__init__()
        self.register_buffer("permutation", torch.randperm(width, generator=generator))
        self.counts = counts
        for count in counts:
            boundaries = torch.arange(count + 1) * width // count
            self.register_buffer(f"boundaries_{count}", boundaries)
        self.projections = nn.ModuleList(
            _make_mlp(count, hidden_dim, intermediate_dim, dropout) for count in counts
        )
        self.scale_attention = nn.Linear(len(counts) * hidden_dim, len(counts))

    def forward(self, features):
        permuted = features.index_select(1, self.permutation)
        prefix = F.pad(permuted.cumsum(dim=1), (1, 0))
        scales = []
        for count, projection in zip(self.counts, self.projections):
            bounds = getattr(self, f"boundaries_{count}")
            boundary_sums = prefix.index_select(1, bounds)
            pooled = boundary_sums[:, 1:] - boundary_sums[:, :-1]
            scales.append(projection(torch.log1p(pooled.clamp_min(0))))
        weights = self.scale_attention(torch.cat(scales, dim=-1)).softmax(dim=-1)
        return (torch.stack(scales, dim=1) * weights.unsqueeze(-1)).sum(dim=1)


class FingerprintEncoder(nn.Module):
    """Gate fingerprint bits and combine scale and fingerprint-type embeddings."""

    def __init__(
        self, fp_dim, hidden_dim, fp_type, encoder_type, intermediate_dim, dropout, use_gate,
    ):
        super().__init__()
        self.reconstruct = encoder_type == "typewise"
        self.gate = FeatureGate(fp_dim) if self.reconstruct and use_gate else None
        if self.reconstruct:
            kinds = MIXED_FINGERPRINT_TYPES if fp_type == "mixed" else (fp_type,)
            self.split_dims = tuple(get_fingerprint_dim(kind) for kind in kinds)
            generator = torch.Generator().manual_seed(0)
            self.parts = nn.ModuleList(
                FingerprintScales(
                    width, BUCKET_COUNTS[kind], hidden_dim, intermediate_dim, dropout, generator,
                )
                for kind, width in zip(kinds, self.split_dims)
            )
            self.type_attention = (
                nn.Linear(len(kinds) * hidden_dim, len(kinds)) if len(kinds) > 1 else None
            )
            self.out_norm = nn.LayerNorm(hidden_dim)
        else:
            self.projection = _make_mlp(fp_dim, hidden_dim, intermediate_dim, dropout)

    def forward(self, fingerprints):
        fingerprints = torch.nan_to_num(fingerprints.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if not self.reconstruct:
            return self.projection(fingerprints)
        if self.gate is not None:
            fingerprints = self.gate(fingerprints)
        parts = fingerprints.split(self.split_dims, dim=-1)
        encoded = [encoder(part) for encoder, part in zip(self.parts, parts)]
        if self.type_attention is None:
            return self.out_norm(encoded[0])
        weights = self.type_attention(torch.cat(encoded, dim=-1)).softmax(dim=-1)
        fused = (torch.stack(encoded, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        return self.out_norm(fused)


class DescriptorEncoder(nn.Module):
    """Condition the fused representation with normalized descriptors and FiLM."""

    def __init__(self, desc_dim, hidden_dim, intermediate_dim, dropout, use_gate):
        super().__init__()
        self.input_norm = nn.LayerNorm(desc_dim)
        self.gate = FeatureGate(desc_dim) if use_gate else None
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scale = nn.Linear(intermediate_dim, hidden_dim)
        self.shift = nn.Linear(intermediate_dim, hidden_dim)
        for layer in (self.scale, self.shift):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, descriptors):
        descriptors = torch.nan_to_num(descriptors.float(), nan=0.0, posinf=0.0, neginf=0.0)
        descriptors = self.input_norm(descriptors)
        if self.gate is not None:
            descriptors = self.gate(descriptors)
        return self.encoder(descriptors)

    def modulate(self, fused, condition):
        return (1 + self.scale(condition)) * fused + self.shift(condition)


class CrossModalAttention(nn.Module):
    """Update a modality using the other modality as the key and value."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context):
        query_token, context_token = query.unsqueeze(1), context.unsqueeze(1)
        update, _ = self.attention(query_token, context_token, context_token, need_weights=False)
        attended = self.attn_norm(query + update.squeeze(1))
        return self.ffn_norm(attended + self.feed_forward(attended))


class ModalTransformerFusion(nn.Module):
    """Aggregate the graph and fingerprint tokens with a CLS token."""

    def __init__(self, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.modality_embed = nn.Parameter(torch.empty(1, 3, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)
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

    def forward(self, tokens):
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        return self.encoder(torch.cat([cls, tokens], dim=1) + self.modality_embed)[:, 0]


class MolecularFusion(nn.Module):
    """Align the graph and fingerprint and preserve their mean in the fusion."""

    def __init__(
        self, hidden_dim, use_fingerprint, method, num_layers, num_heads, dropout,
        use_cross_modal_attn, fusion_residual, residual_gate_init,
    ):
        super().__init__()
        self.method = method
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        if use_fingerprint:
            self.fingerprint_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            )
            if use_cross_modal_attn:
                self.graph_from_fingerprint = CrossModalAttention(hidden_dim, num_heads, dropout)
                self.fingerprint_from_graph = CrossModalAttention(hidden_dim, num_heads, dropout)
            else:
                self.graph_from_fingerprint = None
                self.fingerprint_from_graph = None

            if method == "transformer":
                self.transformer = ModalTransformerFusion(hidden_dim, num_layers, num_heads, dropout)
            elif method == "attention":
                self.token_attention = nn.Linear(2 * hidden_dim, 2)
            elif method == "concat":
                self.concat = _make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
            self.out_norm = nn.LayerNorm(hidden_dim)

        if use_fingerprint and fusion_residual and method != "sum":
            self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        else:
            self.register_parameter("residual_gate", None)

    def forward(self, graph, fingerprint):
        if fingerprint is None:
            return self.graph_projection(graph)

        if self.graph_from_fingerprint is not None:
            graph, fingerprint = (
                self.graph_from_fingerprint(graph, fingerprint),
                self.fingerprint_from_graph(fingerprint, graph),
            )
        tokens = torch.stack(
            [self.graph_projection(graph), self.fingerprint_projection(fingerprint)], dim=1,
        )
        mean = tokens.mean(dim=1)
        if self.method == "transformer":
            fused = self.transformer(tokens)
        elif self.method == "concat":
            fused = self.concat(tokens.flatten(1))
        elif self.method == "attention":
            weights = self.token_attention(tokens.flatten(1)).softmax(dim=-1)
            fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        else:
            fused = mean

        if self.residual_gate is not None:
            gate = self.residual_gate.sigmoid()
            fused = gate * fused + (1 - gate) * mean
        return self.out_norm(fused)
