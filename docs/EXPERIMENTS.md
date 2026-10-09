# Experiment map — one row per published result

Every row gives the paper's number, the config, and the command. Results land in
`results/<name>/`, one JSON per case, and `scripts/aggregate.py` prints ours next to the paper's.

Read `DEVIATIONS.md` before trusting any comparison: a few rows cannot be matched exactly, and the
compute bill for the full set is large.

---

## A. CIFAR-10 core

### A1. Baseline (Table 8 first row, Table 1 last row)
ResNet-18, 1% budget, eps=16, `K=1`, differentiable augmentation, `R=8`, `M=250`, victim augments.
**Paper: 91.25% (±6.14) poison success, 92.20% validation accuracy.**

```bash
python scripts/run_experiment.py --config cifar10/baseline.yaml
```

### A2. Table 1 — differentiable augmentation vs model ensembles
The caption's claim: "Differentiable data augmentation is able to replace a large 8-model ensemble,
without increasing computational effort."

| Config | Ensemble | Diff. aug | Victim augments | Paper |
|---|---|---|---|---|
| `cifar10/table1_K1_noaug_victim_noaug.yaml` | 1 | no | no | 100.00 (±0.00) |
| `cifar10/table1_K1_noaug_victim_aug.yaml` | 1 | no | yes | 32.50 (±12.27) |
| `cifar10/table1_K8_noaug_victim_noaug.yaml` | 8 | no | no | 78.75 (±11.77) |
| `cifar10/table1_K1_diffaug_victim_aug.yaml` | 1 | yes | yes | 91.25 (±6.14) |

```bash
for c in table1_K1_noaug_victim_noaug table1_K1_noaug_victim_aug \
         table1_K8_noaug_victim_noaug table1_K1_diffaug_victim_aug; do
  python scripts/run_experiment.py --config cifar10/$c.yaml
done
python scripts/aggregate.py --table table1
```

### A3. Table 2 — the objective, swapped inside one framework
Only the loss changes; optimizer, restarts, budget, augmentation and evaluation are identical.

| Config | Paper (ConvNet, eps=32) | Paper (ResNet-18, eps=16) |
|---|---|---|
| `table2_*_gm.yaml` (proposed) | 86.25 (±9.43) | 90.00 (±3.87) |
| `table2_*_bullseye.yaml` | 78.75 (±7.66) | 3.75 (±3.56) |
| `table2_*_frogs.yaml` | 52.50 (±12.85) | 1.25 (±1.19) |
| MetaPoison* | 35.00 (±11.01) | 42.50 (±8.33) |

MetaPoison* is not reimplemented here — see `DEVIATIONS.md`. The qualitative result to look for:
feature-collision objectives work on the shallow ConvNet and collapse on the ResNet, because
"they are designed to attack fine-tuning scenarios where the feature extractor is nearly fixed".

### A4. Section 5.1 inset — restarts and ensembles vs brewing time
**Paper: ~0.90 at `R=1,K=1`, rising to ~0.98–0.99 at `K=4..8`; more restarts help independently.**

```bash
python scripts/sweep.py --config cifar10/inset_RK_sweep.yaml --grid restarts=1,8 ensemble=1,2,4,8
python scripts/plot.py sweep --file results/inset_RK_sweep_sweep.json --x ensemble --series restarts
```

### A5. Table 10 — clean validation accuracy
**Paper: 92.13–92.27% unpoisoned, 92.03–92.20% poisoned; the drop is ~0.1%.** The poisoned column is
already in every result file (`mean_valid_acc`); the unpoisoned reference is:

```bash
python -c "import sys; sys.path.insert(0,'src'); \
from witchesbrew.config import load_config; from witchesbrew.evaluate import clean_baseline_accuracy; \
print(clean_baseline_accuracy(load_config('configs/cifar10/baseline.yaml')))"
```

## B. CIFAR-10 extras

### B1. Table 8 — method ablations
| Config | Change | Paper |
|---|---|---|
| `cifar10/table8_cw.yaml` | Carlini-Wagner target loss | 77.50 (±9.32) |
| `cifar10/table8_m50.yaml` | `M=50` instead of 250 | 40.00 (±10.87) |
| `cifar10/table8_euclidean.yaml` | Euclidean gradient distance (eq. 2 directly) | 61.25 (±9.75) |

### B2. Fig. 12 left — thin ResNets, cosine vs Euclidean
**Paper: Euclidean dominates below width 16; cosine is superior for larger widths and "seems to be
able to make use of the greater representative power of the wider networks to find vulnerabilities".**

```bash
python scripts/sweep.py --config cifar10/fig12_width.yaml \
  --grid width=4,8,16,32,48,64,96 objective=cosine,euclidean
python scripts/sweep.py --config cifar10/fig12_width.yaml --grid width=4,8,16,32,48,64,96 \
  --set recipe=poison-frogs
```

### B3. Fig. 12 right — how much pretraining the surrogate needs
**Paper: at eps=16, 20 epochs of pretraining already suffice; at eps=8, success keeps rising to 40.**

```bash
python scripts/sweep.py --config cifar10/fig12_pretrain.yaml \
  --grid pretrain_epochs=0,5,10,20,30,40 eps=8,16
```

### B4. Table 9 — multiple targets
**Paper grid** (targets, budget → success): (1, 1%) 90.00 · (1, 4%) 95.00 · (1, 6%) 90.00 ·
(2, 1%) 65.00 · (4, 1%) 36.25 · (4, 4%) 65.00 · (4, 6%) 46.25 · (5, 5%) 44.00 · (6, 6%) 30.83 ·
(8, 4%) 25.62 · (16, 4%) 18.12.

```bash
python scripts/sweep.py --config cifar10/table9_multitarget.yaml \
  --grid targets=1,2,4,8,16 budget=0.01,0.04,0.06
```

### B5. Table 3 — transfer across architectures
**Paper (benchmark, 1%, eps=8):** `K=1` → 45 / 36 / 8 (ResNet-18 / MobileNet-V2 / VGG11), avg 29.67;
`K=4` → 55 / 37 / 7, avg 33.00; `K=6` heterogeneous → 49 / 38 / 35, avg **40.67**.

```bash
# gray-box diagonal and cross-architecture transfer on the paper's 10 cases
python scripts/sweep.py --config cifar10/transfer_benchmark_k1.yaml \
  --grid victim_net=ResNet18,MobileNetV2,VGG11
# the heterogeneous ensemble that fixes the VGG11 column
python scripts/sweep.py --config cifar10/transfer_benchmark_k6_heterogeneous.yaml \
  --grid victim_net=ResNet18,MobileNetV2,VGG11
```

Note: this runs the *setting* of Table 3 on the paper's 10 published cases, not the benchmark's own
100 scenarios, which live in the Schwarzschild et al. repository. `DEVIATIONS.md` explains the
difference and what it would take to run the benchmark itself.

### B6. Figs. 2, 5, 9, 11 — the analysis plots
Run one case with diagnostics on, for a poisoned and a clean victim:

```bash
python scripts/run_experiment.py --config cifar10/baseline.yaml \
  --set diagnostics=true --set case_ids=0 --set victim_runs=1 --set name=diag_poisoned
python scripts/run_experiment.py --config cifar10/baseline.yaml \
  --set diagnostics=true --set case_ids=0 --set victim_runs=1 --set budget=0.0 --set name=diag_clean
python scripts/plot.py alignment --case results/diag_poisoned/case00.json
python scripts/plot.py target --case results/diag_poisoned/case00.json
```

**Paper (Fig. 2): the poisoned model keeps a strictly positive average batch cosine similarity with
the adversarial gradient; the clean model's is negative.** Fig. 5 plots the eq. 4 bound against the
`a_k L ≈ 1` reference line.

## C. Defenses

### C1. Fig. 4b — differentially private SGD
Clipping fixed at 1, noise swept over {0, 1e-5, 1e-4, 1e-3, 1e-2}, both on the mini-batch level.
**Paper: to push poison success below 15% you give up more than 20% validation accuracy.**

```bash
python scripts/run_defense.py dp --config defenses/dp_sweep.yaml
python scripts/run_defense.py dp --config defenses/dp_sweep_aware.yaml --attacker-aware
python scripts/plot.py dp --file results/dp_sweep_dp_sweep.json
```

### C2. Fig. 4a / Table 6 — outlier filtering
4% budget, eps=16, 6 seeded runs. **Paper: 10% filtering removes 248 poisons / 252 clean against
200 / 300 for random guessing; 20% filtering removes 467 / 533 against 400 / 600.**

```bash
python scripts/run_defense.py filtering --config defenses/filtering.yaml
python scripts/plot.py filtering --file results/filtering_filtering.json
```

The second thing to check here is the absence of feature collision: the mean distance from poisons to
the target feature should be no smaller than from clean images of the same class, which is the
difference from Poison Frogs that Fig. 4a makes.

## D. ImageNet

Needs ILSVRC2012 on disk (`data/README.md`) and a multi-GPU machine; read `DEVIATIONS.md` first.
Brewing uses torchvision pretrained weights and no ensembles (section 3.4, appendix B.2).

| Result | Config | Paper |
|---|---|---|
| Fig. 3 top: budget/eps sweep, ResNet-18 | `imagenet/fig3_budget_eps.yaml` | 0.05% + eps=8 poisons a ResNet-18 ~80% of the time |
| Fig. 3 bottom: architectures at 0.1%, eps=16 | `imagenet/fig3_archs.yaml` | VGG16, ResNet18, MobileNet-v2, ResNet34, ResNet50 all compromised |
| Fig. 13: direct transfer, 0.1%, eps=16 | `imagenet/fig13_transfer.yaml` | diagonal 0.9 / 0.6 / 0.9; ResNet18→ResNet34 0.6 |
| Table 7: victim data ablation, 5 runs | `imagenet/table7_ablation.yaml` | 60% success with 70% of data removed, 100% with 50% |

```bash
python scripts/sweep.py --config imagenet/fig3_budget_eps.yaml \
  --grid budget=0.001,0.0005,0.0001 eps=8,16
python scripts/sweep.py --config imagenet/fig13_transfer.yaml \
  --grid net=ResNet18,MobileNetV2,ResNet34 victim_net=ResNet18,MobileNetV2,ResNet34
# regime 1, victim ablation only (paper: 60% at 70% removed, 100% at 50% removed)
python scripts/sweep.py --config imagenet/table7_ablation.yaml --grid data_ablation=0.3,0.5
# regime 2, the brewing network is trained on a reduced set too (paper: 60% / 80%)
python scripts/sweep.py --config imagenet/table7_ablation.yaml --grid data_ablation=0.3,0.5   --set ablate_surrogate=true
```

Not reproduced: the Cloud AutoML experiment of section 5.2 (`DEVIATIONS.md`).

## E. Verification before any of the above

```bash
pytest tests/                                            # unit tests, seconds, CPU
python scripts/run_experiment.py --config smoke_cpu.yaml  # whole pipeline, minutes, CPU
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export png
```

The smoke config shrinks the model, the steps, the epochs and the training set, so it proves the
pipeline runs — it reproduces no published number, and its result files record that.

`tests/` is the part that does not need a GPU or the dataset: `test_components.py` checks the threat
model, the differentiable augmentation and the objectives; `test_brew.py` checks that Algorithm 1
actually lowers `B` and returns a perturbation inside the eps-ball; `test_pipeline.py` runs the whole
case → brew → victim → result-file path on a synthetic dataset.
