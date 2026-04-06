"""
iResNet100 backbone and ArcFace loss for face recognition.
Rewritten to match InsightFace's arcface_r100.onnx weight layout exactly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """SE block using Conv1x1 to match ONNX [C,1,1] weight layout."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))
        return x * w


class IRBlock(nn.Module):
    """IR block matching ONNX structure:
       BN1 → Conv1 → PReLU → Conv2 → BN2 → SE → + shortcut
       Note: only ONE bn (bn1) before conv1, no bn0.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)          # pre-activation BN (called bn1 in ONNX)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch) if use_se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.bn1(x))
        out = self.prelu(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return out + self.shortcut(x)


def _make_stage(in_ch, out_ch, n_blocks, stride, use_se):
    layers = [IRBlock(in_ch, out_ch, stride=stride, use_se=use_se)]
    for _ in range(1, n_blocks):
        layers.append(IRBlock(out_ch, out_ch, stride=1, use_se=use_se))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────

class IResNet100(nn.Module):
    """iResNet100 matching InsightFace ONNX layout."""

    _STAGE_BLOCKS = (3, 13, 30, 3)

    def __init__(self, embedding_size: int = 512, use_se: bool = True, dropout: float = 0.4):
        super().__init__()
        # Stem — named to match ONNX conv0 + features_stem area
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        # Stages — named layer1..layer4 to match ONNX BN keys
        self.layer1 = _make_stage(64,  64,  self._STAGE_BLOCKS[0], stride=2, use_se=use_se)
        self.layer2 = _make_stage(64,  128, self._STAGE_BLOCKS[1], stride=2, use_se=use_se)
        self.layer3 = _make_stage(128, 256, self._STAGE_BLOCKS[2], stride=2, use_se=use_se)
        self.layer4 = _make_stage(256, 512, self._STAGE_BLOCKS[3], stride=2, use_se=use_se)
        # Output head — named to match ONNX: features (BN), bn2 (BN1d), fc (Linear)
        self.features = nn.BatchNorm2d(512)
        self.dropout  = nn.Dropout(p=dropout)
        self.flatten  = nn.Flatten()
        self.fc       = nn.Linear(512 * 7 * 7, embedding_size, bias=False)
        self.bn2      = nn.BatchNorm1d(embedding_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.features(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x
# ──────────────────────────────────────────────────────────────────────────────
# ArcFace loss head
# ──────────────────────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace, Deng et al., CVPR 2019).

    Learns a class prototype matrix W ∈ ℝ^{C × d}. At each forward pass:
      1. Embeddings and W rows are L2-normalised → dot product = cos θ.
      2. The target-class angle gets the additive margin m: cos(θ + m).
      3. Logits are scaled by s before cross-entropy.

    Parameters
    ----------
    embedding_size : int
        Must match the backbone's output dimension.
    num_classes : int
        Number of identity classes in the training set.
    scale : float
        Feature scale *s* (default 64).
    margin : float
        Additive angular margin *m* in radians (default 0.5 ≈ 28.6°).
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self._cos_m = math.cos(margin)
        self._sin_m = math.sin(margin)
        self._th    = math.cos(math.pi - margin)           # safe-zone boundary
        self._mm    = math.sin(math.pi - margin) * margin  # linear fallback

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb    = F.normalize(embeddings, p=2, dim=1)
        w      = F.normalize(self.weight,  p=2, dim=1)
        cosine = F.linear(emb, w)                           # (N, C)

        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0))
        phi    = cosine * self._cos_m - sine * self._sin_m  # cos(θ + m)
        # Clip for θ > π − m to keep gradients stable
        phi    = torch.where(cosine > self._th, phi, cosine - self._mm)

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        logits  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits  = logits * self.scale

        return F.cross_entropy(logits, labels)
