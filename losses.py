from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def shuffle_batch(x: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2:
        raise ValueError("Mutual-information estimation needs batch size >= 2.")
    return x[torch.randperm(x.size(0), device=x.device)]


def dv_bound(joint_scores: torch.Tensor, marginal_scores: torch.Tensor) -> torch.Tensor:
    joint_flat = joint_scores.flatten(start_dim=1)
    marginal_flat = marginal_scores.flatten(start_dim=1)
    joint_per_sample = joint_flat.sum(dim=1)
    marginal_per_sample = marginal_flat.sum(dim=1)
    e_joint = joint_per_sample.mean()
    log_e_marginal = (
        torch.logsumexp(marginal_per_sample, dim=0)
        - torch.log(torch.tensor(
            float(marginal_per_sample.numel()),
            device=marginal_per_sample.device,
            dtype=marginal_per_sample.dtype,
        ))
    )
    return e_joint - log_e_marginal


def estimate_global_mi(
    statistics_net: nn.Module,
    image_features: torch.Tensor,
    representation: torch.Tensor,
) -> torch.Tensor:
    joint_scores = statistics_net(image_features, representation)
    marginal_scores = statistics_net(image_features, shuffle_batch(representation))
    return dv_bound(joint_scores, marginal_scores)


def estimate_local_mi(
    statistics_net: nn.Module,
    local_features: torch.Tensor,
    representation: torch.Tensor,
) -> torch.Tensor:
    joint_scores = statistics_net(local_features, representation)
    marginal_scores = statistics_net(local_features, shuffle_batch(representation))
    return dv_bound(joint_scores, marginal_scores)


# ── FIX Bug 2 ─────────────────────────────────────────────────────────────────
# Paper (Eq. 10): discriminator treats (E_M, I_M) joint pairs as REAL (1)
# and shuffled (marginal) pairs as FAKE (0).
# Original code had the labels reversed, completely breaking disentanglement.
def discriminator_loss(
    discriminator: nn.Module,
    expression: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    joint_logits    = discriminator(expression.detach(), identity.detach())
    marginal_logits = discriminator(expression.detach(), shuffle_batch(identity.detach()))

    # joint  → REAL  (label 1)   [FIXED: was 0]
    # marginal → FAKE (label 0)  [FIXED: was 1]
    real_joint   = torch.ones_like(joint_logits)
    fake_marginal = torch.zeros_like(marginal_logits)

    return (
        F.binary_cross_entropy_with_logits(joint_logits, real_joint)
        + F.binary_cross_entropy_with_logits(marginal_logits, fake_marginal)
    )


# ── FIX Bug 3 ─────────────────────────────────────────────────────────────────
# The encoder wants to FOOL the discriminator: make joint (E, I) pairs look
# like INDEPENDENT (marginal) samples, i.e. push discriminator output toward 0.
# Original code targeted 1 (real), which pointed the encoder the wrong way.
def encoder_adversarial_loss(
    discriminator: nn.Module,
    expression: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    joint_logits = discriminator(expression, identity)
    # Fool discriminator: make joint pairs look FAKE / independent (label 0)
    # [FIXED: was ones_like, i.e. targeting label 1 = real]
    fool_as_marginal = torch.zeros_like(joint_logits)
    return F.binary_cross_entropy_with_logits(joint_logits, fool_as_marginal)
