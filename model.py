from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


@dataclass
class EncoderOutput:
    embedding: torch.Tensor
    global_features: torch.Tensor
    local_features: torch.Tensor


class ResNet18Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        input_channels: int = 1,
        pretrained_weights: str | Path | None = None,
    ) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)

        if input_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        if pretrained_weights is not None:
            _load_resnet18_backbone_weights(backbone, pretrained_weights, input_channels)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(backbone.fc.in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        local_features  = self.features(x)
        global_features = self.pool(local_features).flatten(1)
        embedding       = self.projector(global_features)
        return EncoderOutput(
            embedding=embedding,
            global_features=global_features,
            local_features=local_features,
        )


class GlobalStatisticsNet(nn.Module):
    def __init__(self, image_dim: int, representation_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim + representation_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image_features: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_features, representation], dim=1))


class LocalStatisticsNet(nn.Module):
    def __init__(self, image_channels: int, representation_dim: int, hidden_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels + representation_dim, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, local_features: torch.Tensor, representation: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = local_features.shape
        representation_map = representation[:, :, None, None].expand(
            batch_size, representation.size(1), height, width
        )
        return self.net(torch.cat([local_features, representation_map], dim=1))


class Discriminator(nn.Module):
    def __init__(self, expression_dim: int = 64, identity_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expression_dim + identity_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, expression: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([expression, identity], dim=1))


class ExpressionClassifier(nn.Module):
    def __init__(self, expression_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.net(expression)


class DICEFER(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        input_channels: int = 1,
        pretrained_resnet: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.expression_encoder = ResNet18Encoder(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained_weights=pretrained_resnet,
        )
        self.identity_encoder = ResNet18Encoder(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained_weights=pretrained_resnet,
        )

        resnet_dim = 512

        # ── FIX Bug 5: separate statistics nets for M-side and N-side ─────────
        # Paper uses U^exp_θM and U^exp_θN (distinct networks for each direction).
        # Sharing one net introduces correlated gradients that degrade MI quality.
        self.exp_global_stats_m = GlobalStatisticsNet(resnet_dim, embedding_dim)
        self.exp_global_stats_n = GlobalStatisticsNet(resnet_dim, embedding_dim)
        self.exp_local_stats_m  = LocalStatisticsNet(resnet_dim, embedding_dim)
        self.exp_local_stats_n  = LocalStatisticsNet(resnet_dim, embedding_dim)

        # Identity stats nets: also separate for M and N
        self.id_global_stats_m = GlobalStatisticsNet(resnet_dim, embedding_dim * 2)
        self.id_global_stats_n = GlobalStatisticsNet(resnet_dim, embedding_dim * 2)
        self.id_local_stats_m  = LocalStatisticsNet(resnet_dim, embedding_dim * 2)
        self.id_local_stats_n  = LocalStatisticsNet(resnet_dim, embedding_dim * 2)

        self.discriminator = Discriminator(embedding_dim, embedding_dim)
        self.classifier    = ExpressionClassifier(embedding_dim, num_classes)

    def encode_expression(self, image: torch.Tensor) -> EncoderOutput:
        return self.expression_encoder(image)

    def encode_identity(self, image: torch.Tensor) -> EncoderOutput:
        return self.identity_encoder(image)

    def classify_expression(self, image: torch.Tensor) -> torch.Tensor:
        expression = self.encode_expression(image).embedding
        return self.classifier(expression)


# ── Weight loading helpers (unchanged) ────────────────────────────────────────

def _load_resnet18_backbone_weights(backbone: models.ResNet, weights_path: str | Path, input_channels: int) -> None:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_common_prefixes(state_dict)

    backbone_state = backbone.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in backbone_state:
            continue
        target = backbone_state[key]
        if key == "conv1.weight" and value.ndim == 4 and target.ndim == 4 and value.shape[1] != target.shape[1]:
            value = _adapt_input_conv_weight(value, target.shape[1])
        if value.shape != target.shape:
            continue
        filtered[key] = value

    if not filtered:
        raise ValueError(
            f"No ResNet-18 backbone tensors from {weights_path} matched torchvision's resnet18. "
            "Pass CASIA-WebFace weights whose keys match conv1/bn1/layer*/fc or a wrapped state_dict."
        )
    backbone.load_state_dict(filtered, strict=False)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported CASIA-WebFace checkpoint format; expected a state_dict-like object.")


def _strip_common_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.", "model.", "backbone.", "encoder.", "resnet.",
        "expression_encoder.features.", "identity_encoder.features.",
    )
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    changed = True
        feature_index_to_name = {
            "0": "conv1", "1": "bn1",
            "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4",
        }
        if normalized.startswith("features."):
            parts = normalized.split(".", 2)
            if len(parts) == 3 and parts[1] in feature_index_to_name:
                normalized = f"{feature_index_to_name[parts[1]]}.{parts[2]}"
        else:
            parts = normalized.split(".", 1)
            if len(parts) == 2 and parts[0] in feature_index_to_name:
                normalized = f"{feature_index_to_name[parts[0]]}.{parts[1]}"
        stripped[normalized] = value
    return stripped


def _adapt_input_conv_weight(weight: torch.Tensor, input_channels: int) -> torch.Tensor:
    if input_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if input_channels > weight.shape[1]:
        repeats = (input_channels + weight.shape[1] - 1) // weight.shape[1]
        return weight.repeat(1, repeats, 1, 1)[:, :input_channels] * (weight.shape[1] / input_channels)
    return weight[:, :input_channels]
