# Deviations — what this repo cannot match exactly, and why

Read this before comparing any number here to the paper.

## 1. Compute

The published protocol is 10 cases × (1 surrogate + brewing + 8 victim trainings of 40 epochs).
That is roughly 90 full CIFAR-10 trainings per table cell. Rough costs on one modern GPU:

| Unit | Cost |
|---|---|
| One CIFAR-10 ResNet-18 training, 40 epochs | ~10–20 min |
| Brewing one case (`R=8`, `M=250`, `K=1`, 500 poisons) | ~5–15 min |
| One case end to end (surrogate cached, 8 victims) | ~1.5–3 h |
| One table cell (10 cases) | ~15–30 h |
| All CIFAR-10 experiments in `EXPERIMENTS.md` | several hundred GPU-hours |

The surrogate is cached per `(architecture, width, pretrain epochs, seed)`, so it is trained once and
reused across restarts and cases that share a seed. Results are written per case, so a sweep can be
stopped and resumed. The paper's own hardware is in appendix B.1 (RTX 2080 Ti, 1 GPU for CIFAR-10).

**This machine has no NVIDIA GPU** (CPU-only PyTorch), so nothing beyond `smoke_cpu.yaml` and the
unit tests was run locally. Real runs belong on Colab or another GPU box; see `notebooks/`.

Any run that used `--set` to shrink the protocol records those overrides in its result JSON and sets
`paper_setting: false`, and `aggregate.py` flags experiments with fewer than 10 cases.

## 2. Results that are not reproduced here

| Paper result | Why not | What it would take |
|---|---|---|
| **MetaPoison\*** column of Table 2 | The paper does not reimplement MetaPoison either — section 5.1 and appendix F.1 use "the original implementation of Huang et al. (2020), but add our larger models". Rewriting it from scratch would be a reimplementation of a different paper, and appendix F.1 notes the numbers also depend on that code's own hyperparameters (it "seems to overfit with eps = 32") | Clone the MetaPoison repository, swap in the ResNet-18/ConvNet of appendix B.2, run it on the same 10 cases |
| **Table 3 exactly** | The benchmark's 100 fixed scenarios and its own victim-training code live in the Schwarzschild et al. (2020) repository, not in this paper. `transfer_benchmark_*.yaml` reproduces the *setting* (1%, eps=8, the three victim architectures) on the paper's 10 published cases, which is a different, smaller sample | Fetch `poison_setups_from_scratch.pickle` from that repository, brew one poison batch per setup, and evaluate with their `benchmark_test.py --from_scratch`. The four baseline rows (Poison Frogs, Convex Polytopes, Clean-Label Backdoor, Hidden-Trigger Backdoor) are quoted by the paper from that benchmark, not computed by it |
| **Cloud AutoML (section 5.2)** | A paid third-party service, and the experiment uploads a poisoned ILSVRC2012 subset to it. Out of scope here | `export_imagefolder` produces the upload layout; the paper's settings were budget 0.1%, eps=32, surrogate ResNet-18, `mobile-high-accuracy-1`, 10 000 milli-node hours, 5 runs |
| **Exact poison image IDs** | Tables 4 and 5 publish the target IDs and seeds but not which training images were perturbed. Our poison subset is drawn from the published seed with NumPy's generator, which does not reproduce the reference code's draw | Not recoverable from the paper. Case-level variance is the reason the paper averages over 10 cases in the first place (appendix B) |
| **ImageNet from scratch** | 40-epoch ILSVRC2012 trainings per victim; the paper used 4 GPUs. Brewing itself is cheap ("in less than four GPU hours") because it only needs pretrained weights | A multi-GPU node and ~150 GB of disk for the dataset |

## 3. Choices where the paper is silent

Fully tabulated in `DESIGN_DECISIONS.md`. The ones that could move a number:

1. **Carlini-Wagner target loss (Table 8 only).** `carlini_wagner` here is the usual margin,
   `max_{j != y_adv} z_j - z_{y_adv}`. The reference code's default `cw_loss` takes the maximum over
   *all* logits including `y_adv`, so its loss and its gradient vanish once the target is already
   classified as `y_adv`; its own `cw_loss2` is the margin used here. Only the Table 8 CW row
   (77.50%) is affected; cross-entropy is the default everywhere else.
2. **Attack step-size schedule.** The paper says "decaying step size" without milestones; the ×0.1
   drops at `M/2.667`, `M/1.6`, `M/1.142` come from the reference code. Table 8 shows the method is
   sensitive to the step budget (`M=50` loses >50 points), so the schedule matters.
3. **`tau` scaling.** The reference scales the signed-Adam learning rate by
   `(poison_batch / 512) / K`. With the paper's own batch of 512 and `K=1` this is a no-op, so it
   only matters for cheaper configurations.
4. **Perturbation initialization.** `randn` scaled into the eps-ball (reference default); the paper
   only says "randomly initialize ... in C".
5. **ConvNet padding.** Inferred as `padding=1` because it is the only reading that yields the
   stated 2304 penultimate width. Unit-tested.
6. **MobileNet-v2 / VGG on CIFAR-10.** "Standard configuration" cannot mean the ImageNet variants on
   32×32 inputs, so a stem adaptation was necessary. Different adaptations change these
   architectures' clean accuracy and hence the transfer numbers of Table 3.
7. **Victim augmentation uses the differentiable module, not `torchvision.RandomCrop`.** The
   reference code does the same, so the attack and the victim draw from one distribution — but that
   distribution uses continuous shifts with bilinear resampling, where a textbook
   `RandomCrop(32, padding=4)` would use integer shifts and no interpolation.
8. **The filtering detector.** Appendix F.2 says "anomaly filtering" without naming a method;
   distance to the poison-class feature centroid is used, because that is the statistic Fig. 4a
   itself plots. A different detector would give different Table 6 counts — though the paper's claim
   is specifically that *no* feature-space detector does much better than guessing.
9. **Batch-norm state while brewing.** The model is in `eval()` mode so poison batches cannot move
   the running statistics, since section 3.3 fixes `theta`. The paper does not state this.

## 4. Scope limits inherited from the paper

Not deviations, but worth restating so nothing here is read as a stronger claim than the paper makes:
only small `l_inf` perturbations are considered (appendix E); the main results use a single target,
and multi-target success is substantially lower at a fixed budget (appendix F.8); ensembles are
dropped for ImageNet (section 3.4); and the gray-box setting assumes the architecture is known,
with transfer measured separately (Table 3, Fig. 13).
