"""Loss functions for the adversarial (target) objective.

Eq. 1 writes the adversarial loss with the same `L` as the training loss, i.e. cross-entropy against
the adversarial label `y_adv`. Appendix F.6 reports a Carlini-Wagner surrogate as an ablation
("while using the Carlini-Wagner loss as a surrogate for cross entropy helped in Huang et al.
(2020), it does not help in our setting"), which is why both are here and cross-entropy is default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(outputs, labels)


def carlini_wagner(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Margin loss: (largest logit over the other classes) - (logit of the intended class)."""
    one_hot = F.one_hot(labels, num_classes=outputs.shape[1]).bool()
    intended = outputs[one_hot]
    other = outputs.masked_fill(one_hot, float("-inf")).max(dim=1).values
    return (other - intended).mean()


def get_target_criterion(name: str):
    key = name.lower()
    if key in {"cross-entropy", "crossentropy", "xent", "ce"}:
        return cross_entropy
    if key in {"carlini-wagner", "cw"}:
        return carlini_wagner
    raise ValueError(f"unknown target criterion: {name}")
