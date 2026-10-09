# Paper notes — Witches' Brew (Geiping, Fowl, Huang, Czaja, Taylor, Moeller, Goldstein; ICLR 2021)

Local copy: `docs/paper/witchesbrew.pdf` (arXiv:2009.02276v2). Section numbers below refer to that PDF.
Everything in this file is taken from the paper; nothing is inferred. Choices that the paper does
*not* fix are collected separately in `DESIGN_DECISIONS.md`.

## 1. What the paper studies

A *targeted, clean-label, from-scratch* data poisoning attack on image classifiers.

- **Targeted** (§1, §2): the goal is that one predetermined test image `x_t` (not in the training set)
  gets classified as an attacker-chosen class `y_adv`. Overall validation accuracy is left intact,
  so the effect is invisible until the target is evaluated. Barreno et al. (2010) call this a
  *poison integrity* attack, as opposed to *poison availability* (degrade accuracy in general),
  which this paper explicitly does not study.
- **Clean-label** (§3.1): the attacker perturbs images but never labels. Perturbations obey
  `||Delta||_inf <= eps` (pixel values in 0..255), so the semantic label of a poisoned image is unchanged
  and it survives cursory human inspection.
- **From-scratch** (§1, §2): the victim trains a *freshly initialized* model on the poisoned data.
  This is harder than the transfer-learning/fine-tuning setting of Poison Frogs (Shafahi et al. 2018)
  and Convex Polytopes (Zhu et al. 2019), where the feature extractor is nearly fixed.

Why it matters (§1, App. A): industrial training sets are scraped, so outsiders can contribute data.
Practitioners surveyed in Kumar et al. (2020) ranked data poisoning the most serious attack on ML
systems. The paper's conclusion for defenders: the training data needs the same scrutiny as the model,
and the only partial mitigation they find is strong differential privacy, at a large accuracy cost.

## 2. Threat model (§3.1)

- Two parties: **attacker** (limited control of training data) and **victim** (trains on that data).
- **Gray-box**: attacker knows the victim's architecture. (Transfer across architectures is measured
  separately in Table 3 and Fig. 13; the Cloud AutoML experiment in §5.2 is fully black-box.)
- Attacker may perturb a fraction of the training set, "usually less than 1%", within an
  `l_inf` bound (e.g. `eps <= 16`).
- Attacker has **no** knowledge of the victim's initialization, mini-batch order, or data augmentation.

## 3. Formal objective

**Bilevel problem (eq. 1).** With `N` training samples `(x_i, y_i)`, of which the first `P` get a
perturbation `Delta_i`, and `T` targets `(x_t_i, y_adv_i)`:

```
min_{Delta in C}  sum_{i=1..T} L( F(x_t_i, theta(Delta)), y_adv_i )
s.t.  theta(Delta) in argmin_theta  (1/N) sum_{i=1..N} L( F(x_i + Delta_i, theta), y_i )
C = { Delta in R^{N x n} : ||Delta||_inf <= eps,  Delta_i = 0 for i > P }
```

Left = *adversarial loss*; right = *training loss*. The paper uses a single target `T = 1` for the
main results (as in Shafahi et al. 2018); multi-target results are in App. F.8 / Table 9.

Direct solution is intractable for deep nets: it needs backpropagation through the whole SGD run
(§2, Muñoz-González et al. 2017). MetaPoison (Huang et al. 2020) approximates it with unrolled
gradient steps; §2 counts that as ~8893 equivalent training epochs with a 24-model ensemble.

**Motivation (§3.2, eq. 2).** An unconstrained attacker would simply insert the target with the wrong
label `y_adv`; then minimizing training loss also minimizes adversarial loss. A clean-label attacker
cannot do that, but can make the poisons *mimic the target's gradient*:

```
grad_theta L(F(x_t, theta), y_adv)  ~=  (1/P) sum_{i=1..P} grad_theta L(F(x_i + Delta_i, theta), y_i)
```

If this holds for every `theta` seen during training, the victim's own descent steps on the training
loss also descend the attacker's adversarial loss.

**Central mechanism (§3.3, eq. 3).** Gradient magnitudes vary hugely across training, so enforcing
eq. 2 for all `theta` is infeasible. Instead *align directions*: take one clean pretrained model with
fixed parameters `theta` and minimize the negative cosine similarity

```
B(Delta, theta) = 1 - < grad_theta L(F(x_t,theta), y_adv) , sum_i grad_theta L(F(x_i+Delta_i,theta), y_i) >
                      -----------------------------------------------------------------------------------
                      || grad_theta L(F(x_t,theta), y_adv) || * || sum_i grad_theta L(F(x_i+Delta_i,theta), y_i) ||
```

Properties the paper stresses:
- **All layers** enter through `theta`, not just the last feature layer as in Poison Frogs.
- One step needs a **single** differentiation of the parameter gradient w.r.t. the input
  (`create_graph=True`), not several unrolled steps as in MetaPoison.
- The loss involves **only the poisoned subset**, so the attack is cheap when the budget is small and
  the attacker does not need the full training set — only the poison subset and one `theta`.
- A **single** parameter vector `theta` suffices, and it is never updated during the attack.
- *Remark*: cosine similarity works best for the models studied; for small/thin models the squared
  Euclidean distance between gradients (i.e. optimizing eq. 2 directly) is stronger — ablation in
  Fig. 12 left.

**Optimizer (§3.3):** signed Adam updates with decaying step size, projecting onto `C` after each step.

## 4. Making it transfer (§3.4) and Algorithm 1

Three techniques, all meant to survive an unknown victim training run:

1. **Differentiable data augmentation and resampling.** At every step of minimizing eq. 3, draw a
   random translation, crop and possibly a horizontal flip per poison, then resample to the original
   resolution with bilinear interpolation, differentiating through the grid sampling operation
   (Jaderberg et al. 2015). Makes the perturbation robust to the victim's augmentation.
2. **Restarts.** Minimize eq. 3 `R` times from random initial perturbations and keep the set with the
   lowest alignment loss `B`. Trades compute for reliability.
3. **Model ensembles.** Attack `K` model instances from different initializations. Expensive —
   "a modest, but stable, increase in performance". Not used for the ImageNet experiments.

Algorithm 1 (paper p. 5), verbatim structure:

```
Require: pretrained clean network F(.,theta); training set (x_i,y_i)_{i=1..N}; target (x_t, y_adv);
         poison budget P < N; bound eps; restarts R; optimization steps M
 1  Select P training images with label y_adv
 2  for r = 1..R:
 3      randomly initialize perturbations Delta^r in C
 4      for j = 1..M:
 5          apply data augmentation to all poisoned samples (x_i + Delta^r_i)_{i=1..P}
 6          compute the average cost B(Delta^r, theta) as in eq. 3 over all poisoned samples
 7          update Delta^r with a step of signed Adam and project onto ||Delta^r||_inf <= eps
 8  choose Delta* as the Delta^r with minimal B(Delta^r, theta)
 9  return poisoned dataset (x_i + Delta*_i, y_i)_{i=1..N}
```

§3.4 adds that lines 6–7 are done in a **stochastic (mini-batch)** setting over the poisons.

Note that poisons are drawn from the class `y_adv` (line 1) — i.e. the class the attacker wants the
target to be assigned to — which is what keeps the attack clean-label.

## 5. Theory (§4, App. C)

**Proposition 1 (Adversarial Descent).** Let `L_adv(theta) = L(F(x_t,theta), y_adv)` be bounded below
with `L`-Lipschitz gradient, and let the victim run gradient descent `theta^{k+1} = theta^k - a_k grad L(theta^k)`.
If for some fixed `beta < 1`

```
a_k L  <  beta (1 - B(Delta, theta^k)) * ||grad L(theta^k)|| / ||grad L_adv(theta^k)||     (eq. 4)
```

then `L_adv(theta^{k+1}) < L_adv(theta^k)`. If additionally `B(Delta, theta^k) < 1 - e` for some
`e > 0` and all `k >= k_0`, then `||grad L_adv(theta^k)|| -> 0` (eq. 5).

Proof = Zoutendijk's classical result (Nocedal & Wright 2006, Thm 3.2) adapted; written out in App. C
via the descent lemma plus the cosine identity.

Reading: alignment makes the training gradient a *descent direction for the adversarial loss*, so a
victim doing ordinary training unwittingly converges toward a stationary point of the attacker's
objective — i.e. it optimizes the original bilevel objective locally.

**Caveat the paper itself raises:** alignment is maximized only for the selected `theta*`, not for
every `theta` on the victim's trajectory. Empirically (Fig. 2) a poisoned ResNet-18 keeps *positive*
average batch cosine similarity between `grad L_adv` and each mini-batch gradient throughout training,
while a clean model's is negative. Fig. 5 plots the eq. 4 bound during an actual poisoned run
(estimating `a_k L ~= 1`). Fig. 11 adds a linear-scale version, a zoom after the first LR drop, and
the same metric for the target's *original* label.

## 6. Experimental methodology (§5, App. B)

- Per experiment: **10 random poison-target cases** (random target class, random poison class, random
  target image, random images to poison). Poisons are computed once per case, then evaluated on
  **8 newly initialized victim models** for CIFAR-10 and **1** for ImageNet.
- *avg. poison success* = fraction of (case, victim) runs where the target is classified as `y_adv`.
  Mere misclassification does not count. Error bars = standard error over the 10 case-level averages.
- Fixed random seeds, reused across methods for fair comparison. Seeds `2000000000-2111111111`
  for the CIFAR-10 experiments and `1000000000-1111111111` for ImageNet; the resulting class pairs and
  target IDs are listed in **Table 4** (CIFAR-10) and **Table 5** (ImageNet). CIFAR target IDs index the
  canonical order of the dataset as downloaded from cs.toronto.edu; ImageNet IDs index the validation
  set with syn-sets in increasing numerical order (torchvision default). Seeds `1000000000-1111111111`
  were used for CIFAR-10 hyperparameter tuning, so the evaluation seeds are different.
- Attack hyperparameters for **all** experiments: `tau = 0.1`, `R = 8`, `M = 250` (§5).
- Victim training is "realistic": data augmentation, SGD with momentum, weight decay, LR drops.

### Models (App. B.2)
- **6-layer ConvNet** (close to Finn et al. 2017 / Krizhevsky et al. 2012): 5 conv layers, kernel 3,
  each followed by ReLU; the last two conv layers followed by max pooling of size 3; then one linear
  layer. Output widths: 64, 128, 128, 256, 256, 2304.
- **CIFAR ResNet-18**: the ImageNet stem is replaced by a 3x3 conv + batch norm + ReLU
  ("effectively equal to upsampling the CIFAR-10 images before feeding them into the model").
- Also used: ResNet-34, MobileNet-v2, VGG-16 "in standard configuration".

### Training recipe (App. B.2)
- ConvNet, MobileNet-v2, VGG-16: initial LR `0.01`. Residual nets: initial LR `0.1`.
- 40 epochs, LR dropped by a factor of 10 at epochs **14, 24, 35**.
- Nesterov SGD, batch size 128, momentum 0.9, weight decay `5e-4`.
- Dataset shuffled every epoch, so where poisons land in mini-batches is random and unknown to the attacker.
- CIFAR-10 augmentation: horizontal flip p=0.5, random crops of size 32x32 with zero-padding of 4.
- ImageNet: resize to 256x256, central crop 224x224; horizontal flip p=0.5; random crops of
  224x224 with zero-padding 28.
- For ImageNet brewing, the clean `theta` is a pretrained model downloaded from torchvision.

### Hardware (App. B.1)
RTX 2080 Ti (1 GPU for CIFAR-10, 4 for ImageNet), some Tesla P100; timings measured on 2080 Ti.

## 7. Results to reproduce

| Result | Setting | Paper numbers |
|---|---|---|
| Inset fig §5.1 | ResNet-18, 1%, eps=16, `R in {1,8}`, `K in {1,2,4,8}`, success vs brewing time | ~0.90 (R=1,K=1) rising to ~0.98-0.99 (K=4..8); restarts also help |
| Table 1 | eps=16, budget 1%: ensemble `K` x diff. data aug. x victim aug. | K=1 no-aug victim-no-aug **100.00 (±0.00)**; K=1 no-aug victim-aug **32.50 (±12.27)**; K=8 no-aug victim-no-aug **78.75 (±11.77)**; K=1 diff-aug victim-aug **91.25 (±6.14)** |
| Table 2 | budget 1%, objective swapped inside the same framework | ConvNet (eps=32): proposed **86.25 (±9.43)**, Bullseye 78.75 (±7.66), Poison Frogs 52.50 (±12.85), MetaPoison* 35.00 (±11.01). ResNet-18 (eps=16): proposed **90.00 (±3.87)**, Bullseye 3.75 (±3.56), Poison Frogs 1.25 (±1.19), MetaPoison* 42.50 (±8.33) |
| Table 3 | Schwarzschild et al. (2020) "Training From Scratch" benchmark, 1%, eps=8, 100 fixed scenarios | ResNet-18 / MobileNet-V2 / VGG11 / avg — K=1: 45 / 36 / 8 / 29.67; K=4: 55 / 37 / 7 / 33.00; K=6 heterogeneous: 49 / 38 / 35 / **40.67**. Prior attacks: Poison Frogs 1.33 avg, Convex Polytopes 0.67, Clean-Label Backdoor 1.00, Hidden-Trigger Backdoor 2.67 |
| Table 8 | ResNet-18, eps=16, 1% ablations | baseline (full diff-aug, R=8, M=250) **91.25 (±6.14)**, val acc 92.20; Carlini-Wagner target loss 77.50 (±9.32); `M=50` 40.00 (±10.87); Euclidean instead of cosine 61.25 (±9.75) |
| Table 9 | multi-target, eps=16 | 1 target/1% 90.00; 1/4% 95.00; 1/6% 90.00; 2/1% 65.00; 4/1% 36.25; 4/4% 65.00; 4/6% 46.25; 5/5% 44.00; 6/6% 30.83; 8/4% 25.62; 16/4% 18.12 |
| Table 10 | clean validation accuracy, eps=16, 1% | unpoisoned 92.13-92.27, poisoned 92.03-92.20 for all `(K,R)`; drop ~0.1% |
| Fig. 12 | left: thin ResNet-18 widths, cosine vs Euclidean vs Poison Frogs; right: success vs number of pretraining epochs for eps=16 and eps=8 | Euclidean dominates below width 16, cosine above; eps=16 works after ~20 pretraining epochs, eps=8 keeps improving to 40 |
| Fig. 4a / Table 6 | CIFAR-10, 4% budget, eps=16, 6 seeds, outlier filtering in feature space | poisons are distributed like clean data; 10% filtering removes 248 poisons / 252 clean (random guessing: 200 / 300); 20% filtering removes 467 / 533 (random: 400 / 600) |
| Fig. 4b / App. F.3 | CIFAR-10, 1%, eps=16, differentially private SGD: gradient clipping fixed at 1, sweeping gradient noise | pushing poison success below 15% costs more than 20% validation accuracy. Attacker-aware variant (noise + clipping also applied to the poison gradient) stays stronger |
| Fig. 3 | ImageNet ResNet-18 from scratch, budget/eps sweep and architectures | 0.05% + eps=8 poisons a ResNet-18 ~80% of the time; extends to MobileNet-v2, ResNet-34, ResNet-50, VGG-16 at 0.1%/eps=16. A ResNet-34 is compromised with 0.1% of data and eps=8 (abstract/§1) |
| Fig. 13 | ImageNet direct transfer, 10 runs, 0.1%, eps=16 | diagonal 0.9/0.6/0.9; ResNet-18 -> ResNet-34 0.6, ResNet-18 -> MobileNet-v2 0.2 |
| Table 7 | ImageNet ablation, 5 runs, victim data removed | victim ablation: 60% (70% removed) / 100% (50% removed); pretrained+victim ablation: 60% / 80% |
| §5.2 Cloud AutoML | poisoned ILSVRC2012 upload, budget 0.1%, eps=32, surrogate ResNet-18 | adversarial label in top-5 in 5/5 runs, top-1 in 1/5 |

Cost comparison the paper makes (§5.2): MetaPoison would need >500 GPU days to poison ImageNet for a
ResNet-18 on their hardware; gradient matching does it in under four GPU hours.

## 8. Defense findings (§5.3, App. F.2, F.3)

- **Sanitization / outlier filtering fails.** Poisoned points are distributed like clean points in
  feature space (Fig. 4a: distance to base-class centroid and to the target feature overlap almost
  perfectly), so filtering is "almost-random guessing" (Table 6). Notably there is *no* feature
  collision as in Poison Frogs — the attack does not need one.
- **Differential privacy partially works, at a price.** Following Hong et al. (2020), gradient noise
  is the key ingredient; the paper fixes gradient clipping at 1 and sweeps noise, applied at the
  *mini-batch* level rather than per example. Below 15% poison success costs >20% validation accuracy
  on CIFAR-10. The paper compares this trade-off to adversarial training against evasion attacks.
  A naive counter-attack (redraw the noise every time `B` is evaluated) is a good baseline.
- Because validation accuracy is unaffected (Table 10) and the labels are clean, neither human
  supervision nor accuracy monitoring detects the attack (App. A). The paper's recommendation is to
  secure the data pipeline itself.

## 9. Scope limits stated by the authors

- Only small `l_inf` perturbations are considered; color transformations, watermarks, other norms and
  perceptual-score hiding are left to prior/other work (App. E).
- Multi-target success does not reach single-target levels at a fixed budget: each target gradient has
  to be matched with a smaller share of the budget (App. F.8).
- Ensembles are costly and were dropped for ImageNet (§3.4).
- One target per attack in the main results; `T = 1` is not a fundamental limitation (§3.1).
