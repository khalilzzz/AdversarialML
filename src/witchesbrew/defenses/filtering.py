"""Data sanitization / outlier filtering as a defense (section 5.3, Fig. 4a, Table 6).

The paper's claim: filtering does not work against this attack, because "poisoned data points are
distributed like clean data points, reducing filtering based methods to almost-random guessing", and
unlike Poison Frogs there is no feature collision to detect. Table 6 quantifies it at a 4% budget
with eps=16, averaged over 6 seeded runs: at 10% filtering, 248 poisons and 252 clean images are
removed, against 200/300 for random guessing.

What this module computes:

* `feature_statistics` -- the two histograms of Fig. 4a: l2 distance to the poison-class feature
  centroid, and l2 distance to the target image's feature, for poisoned and clean images alike.
* `filter_outliers` -- removes the farthest fraction of each class and counts what was caught. The
  paper says "anomaly filtering" without naming the detector; centroid distance is used because it
  is the statistic Fig. 4a itself plots (flagged in DESIGN_DECISIONS.md section 8).
* `random_guessing_baseline` -- the analytic comparison column of Table 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.factory import bypass_last_layer


@dataclass
class FilterReport:
    filtering: float
    poisons_removed: int
    clean_removed: int
    poisons_removed_random: float
    clean_removed_random: float

    def to_dict(self) -> dict:
        return {
            "filtering": self.filtering,
            "poisons_removed": self.poisons_removed,
            "clean_removed": self.clean_removed,
            "poisons_removed_random": self.poisons_removed_random,
            "clean_removed_random": self.clean_removed_random,
        }


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Penultimate-layer features, the space these defenses operate in."""
    feature_model, _ = bypass_last_layer(model)
    model.eval()
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        chunks.append(feature_model(batch).cpu())
    return torch.cat(chunks)


def feature_statistics(
    features: torch.Tensor,
    target_feature: torch.Tensor,
) -> dict[str, np.ndarray]:
    """The two distances plotted in Fig. 4a."""
    centroid = features.mean(dim=0, keepdim=True)
    return {
        "distance_to_centroid": torch.linalg.norm(features - centroid, dim=1).numpy(),
        "distance_to_target": torch.linalg.norm(features - target_feature.view(1, -1), dim=1).numpy(),
    }


def filter_outliers(
    distances: np.ndarray,
    is_poison: np.ndarray,
    fraction: float,
) -> FilterReport:
    """Remove the `fraction` most distant points and count how many were poisons."""
    count = len(distances)
    remove = int(round(fraction * count))
    order = np.argsort(-distances)  # farthest first
    removed = order[:remove]
    poisons_removed = int(is_poison[removed].sum())
    clean_removed = int(remove - poisons_removed)
    total_poisons = int(is_poison.sum())
    return FilterReport(
        filtering=fraction,
        poisons_removed=poisons_removed,
        clean_removed=clean_removed,
        poisons_removed_random=fraction * total_poisons,
        clean_removed_random=fraction * (count - total_poisons),
    )


def random_guessing_baseline(total: int, poisons: int, fraction: float) -> tuple[float, float]:
    """Expected counts for a detector that picks at random -- Table 6's comparison rows."""
    return fraction * poisons, fraction * (total - poisons)
