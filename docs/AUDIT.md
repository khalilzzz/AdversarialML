# Correctness audit

A line-by-line review of the implementation against the paper, plus numerical verification of the
parts that can be checked in closed form. Every defect found is listed with what was wrong, why it
mattered, and the test that now prevents it from coming back.

Run `pytest tests/` (112 tests, CPU, ~13 s) to re-check all of it.

---

## 1. Numerical verification of the method

These are not "the loss went down" checks; they pin the implementation to the equations.

| Property | Verified how | Result |
|---|---|---|
| `B` is exactly eq. 3 | compared against `1 - cos(g_target, g_poison)` built from an independent `autograd.grad` call on a float64 model | agrees to `9e-13` |
| `dB/dDelta` is correct | central finite differences on a single pixel of the perturbation, float64 | relative error `1.3e-07` |
| the second-order path is real | the above only works if `create_graph=True` keeps the parameter gradient differentiable w.r.t. the input (§3.3) | confirmed |
| sum vs. mean reduction | eq. 3 writes a sum, eq. 2 a mean; for the cosine these must coincide | identical to `7e-13` |
| cosine range | `cos(g, g)` gives `B = 0`, `cos(-g, g)` gives `B = 2` | exact |
| eps-ball in pixel units | project a huge perturbation, denormalize, quantize to 8-bit | `max|dx| = 16/255` exactly, image stays in `[0, 1]` |
| Table 6 baseline arithmetic | random-guessing columns at a 4% budget in a 5000-image class | 200/300 and 400/600 — the paper's numbers exactly |
| Fig. 2's cosine == `1 - B` | the diagnostics module and the objective module computed on the same batch | agree to `1e-5` |
| eq. 4's bound | compared against `(1 - B)·‖∇L‖/‖∇L_adv‖` rebuilt by hand | agrees to `1e-4` |

## 2. Defects found and fixed

### 2.1 Restart selection scored the wrong perturbation
`B` was taken from the value accumulated *during* the final optimization step, which belongs to the
perturbation as it was **before** that step's update and projection. Algorithm 1 line 10 selects the
restart with the minimal `B` of the perturbation it returns, so with `R = 8` the chosen set of
poisons could be the wrong one. Now `B` is re-evaluated after the final projection.
*Test:* `test_restart_score_describes_the_returned_perturbation`.

### 2.2 Projection could leave the eps-ball
The eps clip and the valid-pixel-range clip were applied in sequence, so the second could push a
coordinate back outside the first — breaking the threat model of §3.1. Replaced by an explicit
intersection of the two intervals, with both endpoints clamped into the ball first.
*Tests:* `test_projection_enforces_both_constraints`, `test_exported_perturbation_stays_within_eps_in_pixel_units`.

### 2.3 The sampling grid was not an identity
The augmentation grid spanned `[-1, 0.9375]` instead of `[-1, 1]`, so every augmented image was
slightly rescaled even at zero shift. That silently blurred and zoomed both the brewing and the
victim pipeline. Rebuilt as an exact identity grid plus a per-image offset, with the translation
budget taken straight from appendix B.2 (±4 px on CIFAR-10, ±28 on ImageNet) instead of a
reference-code grid constant.
*Tests:* `test_zero_shift_and_no_flip_is_the_identity`, `test_flip_is_a_horizontal_mirror`,
`test_translation_budget_matches_the_paper_padding`.

### 2.4 Diagnostics measured the wrong gradient
Fig. 2 plots the alignment with "the gradient of each mini-batch" of a **poisoned** run. The
measurement iterated the loader directly, so it used clean, unaugmented batches — a different
quantity from the one the victim's optimizer actually follows. The training loop now exposes the
exact batch it feeds the network (poisons added, then augmentation) and the measurement reuses it.
*Test:* `test_prepare_hook_changes_the_measured_batches`.

### 2.5 Diagnostics mutated the model it was measuring
Getting the victim's own gradients requires train mode, which makes batch norm update its running
statistics — so merely observing a run changed it. Buffers are now snapshotted and restored.
*Test:* `test_measurement_does_not_disturb_the_model_being_measured` (also asserts no weight or
`.grad` is touched).

### 2.6 Feature split was wrong for MobileNet-v2
`bypass_last_layer` sliced off the last child and flattened. MobileNet-v2 does its global pooling
inside `forward` rather than in a child module, so the "features" were a `[N, 1280, 4, 4]` map that
no longer matched the classifier — the Poison Frogs and Bullseye objectives would have crashed or
matched the wrong vectors on that architecture. A pooling step is now applied when the tensor is
still spatial, which is a no-op for the architectures that already pool in a child.
*Verified for* ResNet-18/34, ConvNet, VGG-11/16 and MobileNet-v2: features feed the head and produce
10 logits.

### 2.7 Surrogate initialization depended on construction order
`get_surrogate` built the model *before* seeding, so the weights of the `k`-th ensemble member
depended on how many models had been built earlier. Since surrogates are cached by seed, a cached
checkpoint and a freshly trained one could differ. Seeding now happens before construction.
*Test:* `test_surrogate_initialization_depends_only_on_its_seed`.

### 2.8 Table 7's two ablation regimes were not separable
Appendix F.5 distinguishes "victim ablation" (only the victim's training set is reduced) from
"pretrained + victim ablation" (the brewing network too). `data_ablation` propagated to the
surrogate unconditionally, so only the second regime was reachable — and both regimes shared one
cached surrogate. Added `ablate_surrogate`, defaulting to regime 1, and put the ablation into the
checkpoint name.
*Tests:* `test_victim_ablation_does_not_reduce_the_surrogate_training_set`.

### 2.9 Multi-target sampling could return too few targets
Appendix F.8's extra targets were drawn without excluding the one already chosen, then truncated, so
a collision silently produced `targets - 1` targets. Now drawn from the complement, with an explicit
error if the class is too small.
*Test:* `test_multiple_targets_are_requested_when_configured`.

### 2.10 Attack learning rate was scaled by the wrong quantity
The reference scales `tau` by `poison_batch_size / 512`; the code used `min(batch, P) / 512`, so a
1% CIFAR-10 budget (500 poisons, batch 512) ran at `tau = 0.0977` instead of the paper's `0.1`.

### 2.11 Sweeping the architecture kept the previous one's learning rate
Appendix B.2 sets the learning rate from the architecture (0.1 for residual nets, 0.01 for the
ConvNet, MobileNet-v2 and VGG). `variants` wrote the new value straight onto a copy, so
`--grid net=ResNet18,ConvNet` trained the ConvNet at 0.1 — the wrong recipe for every point of the
sweep after the first. Derived defaults are now recomputed, unless the user pinned them explicitly.
*Tests:* `test_sweeping_the_architecture_rederives_its_learning_rate`,
`test_an_explicit_learning_rate_survives_an_architecture_sweep`.

### 2.12 Attacker-aware DP bypassed the objective settings
The counter-attack of appendix F.3 had its own copy of the cosine, so running it with
`objective: euclidean` or `cosine_mode: per_layer` silently ignored both. The hook is now a
parameter of the single objective function. *Exercised for* all five recipe/objective variants.

Also removed: an unused `BrewResult` dataclass, a dead `clip_valid` helper, a misleading
`@torch.no_grad()` decorator on a function that needs gradients, and an unreadable nested ternary in
the model factory.

## 3. Checked and found correct

- **Direction of the attack.** Minimizing `B` maximizes alignment; the victim's descent on the
  training loss then descends `L_adv` — the sign convention of Prop. 1 holds end to end.
- **Clean-label property.** Poisons are drawn from the adversarial class and keep their own true
  labels; only pixels change. *Test:* `test_poisons_are_drawn_from_the_adversarial_class`.
- **Published cases.** Tables 4 and 5 are used verbatim; all ten CIFAR-10 pairs have distinct target
  and poison classes, and target IDs are valid validation indices.
- **Budget arithmetic.** `ceil(budget × N)`: 1% → 500 poisons, 4% → 2000.
- **Victim recipe.** Nesterov SGD, batch 128, momentum 0.9, weight decay 5e-4, 40 epochs, ×0.1 at
  epochs 14/24/35 — traced through `MultiStepLR` to confirm epoch 14 is the first reduced epoch.
  LR 0.1 for residual nets, 0.01 otherwise.
- **ConvNet shape.** 32 → 10 → 3 with `padding=1` and two stride-3 pools gives `256·3·3 = 2304`,
  the width appendix B.2 states. *Test:* `test_convnet_penultimate_width_matches_the_paper`.
- **CIFAR ResNet stem.** 3×3, stride 1, no max-pool.
- **Attack step schedule.** `M = 250` → drops at steps 93, 156, 218.
- **One optimizer step per pass** over the poison mini-batches, matching the reference.
- **Signed Adam** replaces the gradient by its sign before the Adam step; projection afterwards.
- **Success criterion.** Only classification as `y_adv` at the end of training counts.
- **Statistics.** Mean over cases of the per-case victim average, standard error with `ddof=1`.
- **Clean accuracy** excludes the target images (appendix F.9).
- **DP defense.** Mini-batch clipping at 1, Gaussian noise of std `clip × noise`; verified that the
  gradient is clipped and then dominated by noise at the high end of the Fig. 4b sweep.
- **Config integrity.** All 29 shipped configs load, and the ones naming a paper result carry that
  result's setting (eps per architecture in Table 2, 4% budget for filtering, `K = 6` heterogeneous
  ensemble for Table 3, and so on).

## 4. Known-imperfect, by choice

Carried in `DEVIATIONS.md`, repeated here because they bound what a match can mean: the per-layer vs
global cosine reading of eq. 3; the attack step-size schedule and perturbation initialization taken
from the reference; the victim's augmentation using continuous shifts with bilinear resampling
rather than integer crops (as the reference does); the CIFAR adaptations of MobileNet-v2 and VGG;
the filtering detector; and the poison image IDs, which the paper never published.
