"""
iResNet100 backbone and ArcFace loss for face recognition.

Architecture matches InsightFace's arcface_r100.onnx (IR-SE-100,
trained on WebFace600K):
  - Stem: Conv3×3 → BN → PReLU
  - 4 stages with block counts [3, 13, 30, 3]  (≈ 100 layers total)
  - SE channel-attention in every residual block
  - Output head: BN → Dropout → Flatten → FC(512) → BN1d

Input:  float32 NCHW, pixel values in [-1, 1], spatial size 112×112.
Output: float32 (N, embedding_size), raw — caller must L2-normalise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class IRBlock(nn.Module):
    """Improved Residual Block (pre-activation BN, PReLU, optional SE).

    Differs from a standard ResNet block in two ways:
      1. BN is applied *before* the first convolution (pre-activation style).
      2. Activation is PReLU (learnable slope) instead of ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.bn0   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se: nn.Module = SEBlock(out_ch) if use_se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.bn0(x))
        out = self.act(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return out + self.shortcut(x)


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int, use_se: bool) -> nn.Sequential:
    layers = [IRBlock(in_ch, out_ch, stride=stride, use_se=use_se)]
    for _ in range(1, n_blocks):
        layers.append(IRBlock(out_ch, out_ch, stride=1, use_se=use_se))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────

class IResNet100(nn.Module):
    """iResNet100 (IR-SE-100) backbone for face recognition.

    Stage block counts: [3, 13, 30, 3]  — total ≈ 100 layers.

    Parameters
    ----------
    embedding_size : int
        Dimension of the output embedding vector (default 512).
    use_se : bool
        Enable Squeeze-and-Excitation blocks (default True).
    dropout : float
        Dropout probability in the output head (default 0.4).
    """

    _STAGE_BLOCKS = (3, 13, 30, 3)

    def __init__(self, embedding_size: int = 512, use_se: bool = True, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.stage1 = _make_stage(64,  64,  self._STAGE_BLOCKS[0], stride=2, use_se=use_se)
        self.stage2 = _make_stage(64,  128, self._STAGE_BLOCKS[1], stride=2, use_se=use_se)
        self.stage3 = _make_stage(128, 256, self._STAGE_BLOCKS[2], stride=2, use_se=use_se)
        self.stage4 = _make_stage(256, 512, self._STAGE_BLOCKS[3], stride=2, use_se=use_se)
        self.output_head = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(p=dropout),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )
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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.output_head(x)


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
