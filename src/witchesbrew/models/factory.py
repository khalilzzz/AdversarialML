"""Model construction by name.

CIFAR-10: the ResNets and the ConvNet come from this package (appendix B.2 describes both). VGG and
MobileNet-v2 are the torchvision architectures with the usual CIFAR adaptation -- the paper says
"standard configuration" but the ImageNet versions downsample a 32x32 input to nothing, so some
adaptation is unavoidable; it is flagged as an assumption in DESIGN_DECISIONS.md section 6.

ImageNet: the torchvision architectures unchanged, with optional pretrained weights, which is what
appendix B.2 prescribes for brewing on ImageNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvmodels

from .convnet import convnet
from .resnet import resnet18, resnet34


def _cifar_vgg(name: str, num_classes: int) -> nn.Module:
    model = getattr(tvmodels, name)(weights=None, num_classes=num_classes)
    # Adapt the classifier head to a 1x1 spatial map (32 -> 1 after five stride-2 pools).
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def _cifar_mobilenet_v2(num_classes: int) -> nn.Module:
    model = tvmodels.mobilenet_v2(weights=None, num_classes=num_classes)
    # Keep resolution: the ImageNet stem has stride 2, which is too aggressive for 32x32.
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[2].conv[1][0].stride = (1, 1)
    return model


def build_model(
    name: str,
    dataset: str = "CIFAR10",
    num_classes: int = 10,
    width: int = 64,
    pretrained: bool = False,
) -> nn.Module:
    key = name.lower().replace("-", "").replace("_", "")
    cifar = dataset.upper() == "CIFAR10"

    if cifar:
        if key == "resnet18":
            return resnet18(num_classes=num_classes, width=width)
        if key == "resnet34":
            return resnet34(num_classes=num_classes, width=width)
        if key in {"convnet", "convnet6", "6layerconvnet"}:
            return convnet(num_classes=num_classes)
        if key in {"vgg11", "vgg16"}:
            return _cifar_vgg(key, num_classes)
        if key in {"mobilenetv2", "mobilenet"}:
            return _cifar_mobilenet_v2(num_classes)
        raise ValueError(f"unknown CIFAR-10 architecture: {name}")

    # ImageNet: torchvision in standard configuration (appendix B.2).
    tv_names = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet": "mobilenet_v2",
        "vgg16": "vgg16",
        "vgg11": "vgg11",
    }
    if key not in tv_names:
        raise ValueError(f"unknown ImageNet architecture: {name}")
    builder = getattr(tvmodels, tv_names[key])
    if pretrained:
        # Appendix B.2: "we download the respective pretrained model from torchvision". The class
        # count is fixed at 1000 by those weights.
        return builder(weights="DEFAULT")
    return builder(weights=None, num_classes=num_classes)


class _PoolAndFlatten(nn.Module):
    """Reduce a feature map to the vector the classification head expects.

    Some architectures (MobileNet-v2) do their global pooling inside `forward` rather than in a
    child module, so slicing off the last child leaves a `[N, C, H, W]` tensor that no longer
    matches the head. Pooling here when the tensor is still spatial reproduces what the model's own
    `forward` does, and is a no-op for the architectures that already pool in a child module.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


def bypass_last_layer(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Split a model into (feature extractor, classification head).

    Needed only by the Poison Frogs and Bullseye objectives, which operate on penultimate-layer
    features. The split takes every top-level child except the last; this is the reference
    implementation's heuristic, and it is a heuristic -- see DESIGN_DECISIONS.md section 7.
    """
    children = list(model.children())
    head = children[-1]
    feature_model = nn.Sequential(*children[:-1], _PoolAndFlatten())
    return feature_model, head


def feature_dim(model: nn.Module, dataset: str, device: torch.device) -> int:
    size = 32 if dataset.upper() == "CIFAR10" else 224
    feature_model, _ = bypass_last_layer(model)
    with torch.no_grad():
        probe = torch.zeros(1, 3, size, size, device=device)
        return int(feature_model(probe).shape[1])
