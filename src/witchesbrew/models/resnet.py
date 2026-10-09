"""ResNets for CIFAR-10, following appendix B.2.

"We make the customary changes to the model architecture for CIFAR-10, replacing the stem of the
original model (which requires ImageNet-sized images) by a convolutional layer of size 3, following
by batch normalization and a ReLU." -- so: 3x3 stem, no stride-2 and no max-pool in the stem; the
rest is the standard He et al. (2015) BasicBlock stack.

`width` exists for the Fig. 12-left ablation, where "the usual base width of 64 is replaced by the
width value shown on the x-axis".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(
        self,
        blocks: list[int],
        num_classes: int = 10,
        width: int = 64,
        block: type[nn.Module] = BasicBlock,
    ) -> None:
        super().__init__()
        self.in_planes = width
        # Stem: 3x3 conv + BN + ReLU, no downsampling (appendix B.2).
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        widths = [width, width * 2, width * 4, width * 8]
        strides = [1, 2, 2, 2]
        layers = []
        for planes, count, stride in zip(widths, blocks, strides):
            layers.append(self._make_layer(block, planes, count, stride))
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(widths[-1] * block.expansion, num_classes)

    def _make_layer(self, block: type[nn.Module], planes: int, count: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (count - 1)
        blocks = []
        for s in strides:
            blocks.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layers(out)
        out = self.flatten(self.pool(out))
        return self.linear(out)


def resnet18(num_classes: int = 10, width: int = 64) -> ResNet:
    return ResNet([2, 2, 2, 2], num_classes=num_classes, width=width)


def resnet34(num_classes: int = 10, width: int = 64) -> ResNet:
    return ResNet([3, 4, 6, 3], num_classes=num_classes, width=width)
