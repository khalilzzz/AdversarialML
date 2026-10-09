"""The paper's "6-layer ConvNet" (appendix B.2), used for the Table 2 comparison.

Verbatim description: "we consider an architecture of 5 convolutional layers (with kernel size 3),
followed by a linear layer. All convolutional layers are followed by a ReLU activation. The last two
convolutional layers are followed by max pooling with size 3. The output widths of these layers are
given by 64, 128, 128, 256, 256, 2304."

The only thing left open is the padding and the pooling stride. With `padding=1` and two pooling
layers of size and stride 3, a 32x32 input becomes 10x10 then 3x3, so the flattened penultimate
width is 256 * 3 * 3 = 2304 -- exactly the stated final width. That is the reading implemented here
(see DESIGN_DECISIONS.md section 6).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        widths = [64, 128, 128, 256, 256]
        layers: list[nn.Module] = []
        previous = in_channels
        for index, width in enumerate(widths):
            layers.append(nn.Conv2d(previous, width, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if index >= len(widths) - 2:  # the last two conv layers are followed by max pooling
                layers.append(nn.MaxPool2d(kernel_size=3, stride=3))
            previous = width
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2304, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(self.features(x)))


def convnet(num_classes: int = 10) -> ConvNet:
    return ConvNet(num_classes=num_classes)
