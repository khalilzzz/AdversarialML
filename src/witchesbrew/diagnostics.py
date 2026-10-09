"""Measurements that support the paper's analysis section rather than its headline numbers.

* **Fig. 2 / 11a** -- per-epoch average over mini-batches of the cosine similarity between the
  adversarial gradient and each mini-batch's training gradient, for a poisoned and a clean model.
  The paper's point: "Crucially, the gradient alignment is strictly positive" for the poisoned model
  while a clean model's angles are negative, i.e. normal training on clean data *increases*
  adversarial loss.
* **Fig. 11c** -- the same quantity computed with the target's original label, the opposite metric.
* **Fig. 5** -- the eq. 4 bound during training. The quantity plotted is
  `(1 - B(Delta, theta_k)) * ||grad L(theta_k)|| / ||grad L_adv(theta_k)||`, compared against the
  practical lower-bound estimate `a_k L ~= 1` ("classical gradient descent converges only if
  a_k L < 1, so we can find an upper bound to this value by 1").
* **Fig. 9** -- target loss and accuracy under the adversarial and the original label, which
  `train.TrainHistory` already records per epoch.
"""

from __future__ import annotations

import torch

from .attacks.losses import get_target_criterion


def _flat_grad(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


def adversarial_gradient(
    model: torch.nn.Module,
    target_inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion_name: str = "cross-entropy",
) -> torch.Tensor:
    criterion = get_target_criterion(criterion_name)
    was_training = model.training
    model.eval()
    loss = criterion(model(target_inputs), labels)
    grad = _flat_grad(model, loss)
    if was_training:
        model.train()
    return grad.detach()


def epoch_alignment(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    target_inputs: torch.Tensor,
    adv_labels: torch.Tensor,
    true_labels: torch.Tensor | None = None,
    max_batches: int | None = None,
    prepare=None,
) -> dict[str, float]:
    """Average batch cosine similarity for one epoch, plus the eq. 4 bound of Fig. 5."""
    criterion = torch.nn.CrossEntropyLoss()
    adv_grad = adversarial_gradient(model, target_inputs, adv_labels)
    adv_norm = adv_grad.norm()
    orig_grad = (
        adversarial_gradient(model, target_inputs, true_labels) if true_labels is not None else None
    )

    cosines, cosines_orig, bounds = [], [], []
    # The mini-batch gradients are the ones the victim would take, so the model has to be in train
    # mode. That makes batch-norm update its running statistics, which would change the very model
    # being measured, so the buffers are snapshotted and restored around the measurement.
    buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    model.train()
    for batch_index, (inputs, labels, indices) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        if prepare is not None:
            inputs = prepare(inputs, indices)
        batch_grad = _flat_grad(model, criterion(model(inputs), labels)).detach()
        batch_norm = batch_grad.norm()
        similarity = float(torch.dot(batch_grad, adv_grad) / (batch_norm * adv_norm + 1e-12))
        cosines.append(similarity)
        # eq. 4 with B = 1 - similarity, so (1 - B) = similarity.
        bounds.append(float(similarity * batch_norm / (adv_norm + 1e-12)))
        if orig_grad is not None:
            cosines_orig.append(
                float(torch.dot(batch_grad, orig_grad) / (batch_norm * orig_grad.norm() + 1e-12))
            )

    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(buffers[name])

    out = {
        "cosine_adv": sum(cosines) / max(1, len(cosines)),
        "bound_eq4": sum(bounds) / max(1, len(bounds)),
    }
    if cosines_orig:
        out["cosine_orig"] = sum(cosines_orig) / max(1, len(cosines_orig))
    return out


def make_diagnostics_fn(device, target_inputs, adv_labels, true_labels, max_batches=None):
    """Adapter matching `train.train_model(diagnostics_fn=...)`."""

    def diagnostics(model, loader, epoch, prepare=None):
        stats = epoch_alignment(
            model, loader, device, target_inputs, adv_labels, true_labels, max_batches, prepare
        )
        stats["epoch"] = epoch
        return stats

    return diagnostics
