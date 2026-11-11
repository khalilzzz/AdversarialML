# Witches' Brew — a reproduction

A from-scratch implementation of **"Witches' Brew: Industrial Scale Data Poisoning via Gradient
Matching"** (Geiping, Fowl, Huang, Czaja, Taylor, Moeller, Goldstein; ICLR 2021), built to reproduce
the paper's CIFAR-10 and ImageNet results and its defense analysis.

The paper is in `docs/paper/witchesbrew.pdf`. Nothing here comes from anywhere else: every
hyperparameter is either stated in the paper, or — where the paper is silent — taken from the
reference implementation the paper itself links in section 5, and flagged as such. The full ledger is
`docs/DESIGN_DECISIONS.md`.

This is defensive/academic work: the paper's own conclusion is that data sanitization fails against
this attack and that only strong differential privacy helps, at a large accuracy cost. Both defenses
are implemented here (`src/witchesbrew/defenses/`) so the claim can be checked rather than taken on
faith.

## What the attack does

A *targeted, clean-label, from-scratch* poisoning attack. The attacker perturbs a small fraction of
the training set (≤1%) within an `l_inf` bound (eps ≤ 16 pixel values), never touching any label,
and the victim then trains a freshly initialized model on that data. One predetermined test image
ends up classified as the attacker's chosen class; overall validation accuracy drops by ~0.1%, so
nothing looks wrong until the target is evaluated.

The mechanism is one equation. Instead of solving the bilevel problem (eq. 1) by unrolling training,
align the poisons' training gradient with the gradient of the adversarial loss on a *single* fixed
pretrained model — minimize

```
B(Delta, theta) = 1 - cos( grad_theta L(x_target, y_adversarial),
                           sum_i grad_theta L(x_i + Delta_i, y_i) )
```

with signed Adam, projecting back into the eps-ball each step. If the training gradient points along
the adversarial gradient, the victim's own descent steps minimize the attacker's objective
(Proposition 1). One step costs a single differentiation of the parameter gradient with respect to
the input, which is what makes the method cheap enough to scale.

## Layout

```
docs/
  GUIDE.md              start here: the two codebases, what validating the paper means, what to run first
  CUSTOM_DATA.md        poison your own ImageFolder dataset, your own target, your own parameters
  PAPER_NOTES.md        the paper, summarized: threat model, eqs. 1-4, Alg. 1, Prop. 1, every result
  DESIGN_DECISIONS.md   every choice -> value -> source (paper section / reference code / assumption)
  EXPERIMENTS.md        one row per published number: config, command, what to expect
  DEVIATIONS.md         what cannot be matched, and the compute bill
  AUDIT.md              correctness audit: what was verified numerically, what was found broken
configs/                base.yaml (paper defaults) + one config per table/figure
src/witchesbrew/
  attacks/brew.py       Algorithm 1
  attacks/objectives.py eq. 3 (cosine), eq. 2 (Euclidean), Poison Frogs, Bullseye
  data/augment.py       differentiable augmentation (section 3.4)
  data/setup.py         the published poison-target cases (Tables 4, 5)
  models/               CIFAR ResNets with the appendix B.2 stem, the 6-layer ConvNet, factory
  train.py              victim/surrogate training (appendix B.2) + the DP defense
  evaluate.py           the appendix B protocol: 10 cases x 8 victims, mean +- standard error
  defenses/             differential privacy (Fig. 4b), outlier filtering (Fig. 4a, Table 6)
  diagnostics.py        gradient alignment (Figs. 2, 11), the eq. 4 bound (Fig. 5)
scripts/                run_experiment, sweep, brew_case, run_defense, aggregate, plot, poison_custom
tests/
  test_components.py    threat model, augmentation, objectives, architectures, config
  test_brew.py          Algorithm 1 really lowers B and respects the eps bound
  test_pipeline.py      cases, budget, poison injection, protocol, resumable results
  test_diagnostics.py   Fig. 2 / Fig. 5 measure what the paper defines, without side effects
  test_configs.py       every shipped config loads and matches its paper setting
notebooks/              Colab runner
```

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
pytest tests/             # 112 tests, CPU, ~13 s
```

A GPU is effectively required for anything beyond the smoke test; see `docs/DEVIATIONS.md` for the
cost table. CIFAR-10 downloads itself; ImageNet must be laid out by hand (`data/README.md`).

## Quickstart

```bash
# 1. whole pipeline on CPU in minutes -- proves it runs, reproduces no published number
python scripts/run_experiment.py --config smoke_cpu.yaml

# 2. the paper baseline: ResNet-18, 1% budget, eps=16, R=8, M=250, 10 cases x 8 victims
python scripts/run_experiment.py --config cifar10/baseline.yaml

# 3. brew one case and look at the poisons (Figs. 6-8 in the paper)
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export png

# 4. compare everything computed so far against the published numbers
python scripts/aggregate.py
```

Results are written per case to `results/<name>/caseNN.json`, so an interrupted run resumes. Any
`--set key=value` override is recorded in the result file and marks the run as *not* the published
protocol.

## Headline numbers to reproduce

| Setting | Paper |
|---|---|
| ResNet-18, CIFAR-10, 1% budget, eps=16 (baseline) | **91.25%** (±6.14) poison success, 92.20% clean accuracy |
| Same, without differentiable augmentation, victim augments | 32.50% (±12.27) — augmentation is what makes it work |
| Poison Frogs objective, same framework, ResNet-18 | 1.25% — feature collision fails from scratch |
| Bullseye objective, same framework, ResNet-18 | 3.75% |
| CIFAR-10 benchmark (1%, eps=8), K=6 heterogeneous ensemble | 40.67% average across three architectures |
| ImageNet ResNet-18, 0.1% budget, eps=8 | reliably compromised, brewed in <4 GPU hours |
| Outlier filtering at 10% (4% budget) | removes 248 poisons / 252 clean vs 200 / 300 for guessing |
| Differential privacy | poison success <15% costs >20% validation accuracy |

Full map with commands: `docs/EXPERIMENTS.md`.

## Reading order

1. `docs/GUIDE.md` — the two codebases side by side, and what to run in which order.
2. `docs/PAPER_NOTES.md` — what the paper claims and how it measures it.
3. `src/witchesbrew/attacks/objectives.py` and `attacks/brew.py` — the method, ~250 lines.
4. `docs/DESIGN_DECISIONS.md` — before changing any hyperparameter.
5. `docs/DEVIATIONS.md` — before comparing any number to the paper.
6. `docs/AUDIT.md` — what has been verified against the equations, and how.

## Citation

```bibtex
@inproceedings{geiping2021witches,
  title     = {Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching},
  author    = {Geiping, Jonas and Fowl, Liam and Huang, W. Ronny and Czaja, Wojciech and
               Taylor, Gavin and Moeller, Michael and Goldstein, Tom},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}
```

Reference implementation cited by the paper: `github.com/JonasGeiping/poisoning-gradient-matching`.
