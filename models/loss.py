"""Query-set focal loss for binary molecular property tasks."""

import torch
import torch.nn.functional as F


def focal_loss(logits, labels, gamma):
    cross_entropy = F.cross_entropy(logits, labels, reduction="none")
    probability_error = -torch.expm1(-cross_entropy)
    weight = probability_error.clamp_min(torch.finfo(cross_entropy.dtype).tiny).pow(gamma)
    return (weight * cross_entropy).mean()
