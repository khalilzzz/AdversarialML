# Sonde fréquentielle — où Witches' Brew dépense son budget (piste 1)

**Extension de recherche, hors-article.** Ce module ne fait pas partie de la reproduction du papier.
Il teste une question ouverte sur *pourquoi* les défenses par purification battent Witches' Brew, et
propose une attaque qui y résiste. Tout est derrière des drapeaux à défaut neutre (`delta_param:
full`, `purify: none`, `brew_purify: none`) : le chemin du papier et les 135 tests restent intacts.

---

## La thèse

Hong, Carlini et Kurakin (ICML 2024) montrent que le simple débruitage bat les attaques de
poisoning, mais concluent que les attaques actuelles sont **faibles** : elles n'exploitent pas le
budget `ε = 16/255`. Leur certificat est d'ailleurs vide à ce budget — le Corollaire 3.4 exige
`√(Σ‖δᵢ‖₂²) ≲ 0,6`, alors que l'attaque réelle vaut `√500 × 3,48 ≈ 78`. La défense marche donc
empiriquement ~100× au-delà de ce qu'elle garantit : il y a un mécanisme non expliqué.

Hypothèse : **WB dépense son budget `ℓ∞` dans un sous-espace — hautes fréquences, hors de la variété
des images — que la purification efface.** PureGen rapporte que JPEG seul ramène WB à 0 % : JPEG est
un filtre passe-bas quantifié, donc si WB y meurt, c'est que ses perturbations vivent dans les hautes
fréquences DCT.

Quatre expériences :

- **E0 — diagnostic** : mesurer combien de budget WB perd dans le sous-espace coupé.
- **E1 — WB-LF** : contraindre `δ` au sous-espace conservé (paramétrisation spectrale), et voir s'il survit.
- **E2 — WB-Ψ** : brasser *à travers* Ψ (BPDA pour JPEG, EOT pour les Ψ stochastiques), puis évaluer
  sous une défense de purification côté victime. L'attaque matche le gradient que la victime purifiée
  produit réellement.
- **E3 — compromis** : tracer succès vs détectabilité (énergie hors-variété) et vs visibilité
  (1 − SSIM), en fonction de `ε` et de la coupure fréquentielle. La courbe est la contribution.

---

## Le résultat linéaire (démontrable au tableau)

Modèle linéaire `f(x) = wᵀx`, perte logistique. Alors `∇_w ℓ(x, y) = r(x, y)·x` avec `r` scalaire :
la direction du gradient d'un poison **est** celle de `xᵢ + δᵢ`, et celle de la cible est `x_t`.

Si le défenseur applique un projecteur orthogonal `P` sur un sous-espace `S` (passe-bas, PCA, …),
l'alignement atteignable vérifie, pour tout `δᵢ` :

```
cos(P(xᵢ+δᵢ), x_t) = ⟨P(xᵢ+δᵢ), P x_t⟩ / (‖P(xᵢ+δᵢ)‖ · ‖x_t‖)  ≤  ‖P x_t‖ / ‖x_t‖  =: ρ_S(x_t)
```

Conséquences :
1. la composante de `δᵢ` dans `S⊥` est **entièrement perdue** ;
2. l'alignement maximal est borné par la fraction d'énergie de la **cible** dans `S` (`ρ_S`) ;
3. à budget `ℓ∞` fixé, l'attaquant optimal cherche `δᵢ ∈ S ∩ {‖δ‖∞ ≤ ε}` — par projections alternées
   (Dykstra), ce que `WB-LF` implémente.

Pour un réseau non linéaire, la règle de la chaîne donne la version au premier ordre :
`∇_θ ℓ(θ; Ψ(x+δ))` ne dépend de `δ` qu'à travers `J_Ψ(x)·δ`, donc toute composante de `δ` dans le
noyau de la jacobienne de la purification est gaspillée.

`ρ_S(x_t)` est mesuré par `rho_subspace` ; c'est la colonne `rho_target` du diagnostic.

---

## Les opérateurs Ψ (synthétiques, aucun modèle externe)

`src/witchesbrew/defenses/purification.py`, tous en espace pixel `[0,1]` :

| Ψ | Ce que c'est | Statut |
|---|---|---|
| `dct` | troncature DCT passe-bas : garde les `cutoff×cutoff` coefficients haut-gauche. **Le projecteur `P` exact** de la Proposition. | exact |
| `jpeg` | DCT par blocs 8×8, quantification (table luminance standard × `quality`), déquant, IDCT. Le filtre de PureGen. Quantification non différentiable → variante straight-through (BPDA) pour l'usage gradient. | exact (quantification réelle) |
| `gaussian` | moyenne EOT de `blur(x + σ·η)` : approximation du débruiteur optimal à un pas (Tweedie : `P` ≈ projection sur le plan tangent local de la variété). | **approximation**, pas un DDPM |

E2 est implémenté avec ces Ψ synthétiques (JPEG BPDA, DCT, débruiteur gaussien un pas). Le vrai DDPM
CIFAR-10 de Hong et al. n'est pas implémenté ; `_psi_by_kind` / `build_purifier` sont le point
d'extension pour l'y brancher.

---

## Les métriques (E0)

`src/witchesbrew/analysis/spectrum.py`. Une perturbation normalisée `δ` est ramenée en pixels par
`δ · ds` (multiplication par l'écart-type, sans ajout de moyenne : `δ` est une différence).

| Métrique | Définition | Lecture |
|---|---|---|
| `radial_dct_profile` | énergie DCT de `δ` binnée par fréquence radiale | la thèse prédit l'énergie en haut du spectre |
| `budget_survival` | `‖Ψ(x+δ) − Ψ(x)‖² / ‖δ‖²` | fraction du budget qui passe Ψ ; `1 −` = budget détruit |
| `alignment_survival` | l'objectif `B = 1 − cos(g_cible, g_poison)` avant et après Ψ | si `B` remonte vers 1 après Ψ, le signal de l'attaque est effacé |
| `rho_subspace` | `‖Ψ(x_t)‖ / ‖x_t‖` | borne de l'alignement atteignable (fraction d'énergie cible conservée) |

**`budget_survival` utilise des nombres aléatoires communs** : pour un Ψ stochastique (gaussian), les
deux évaluations partagent la même réalisation de bruit, sinon la variance du bruit écrase `δ` et le
ratio explose. Le ratio peut légèrement dépasser 1 pour un Ψ non linéaire (la quantification JPEG
peut pousser un coefficient au-delà d'une borne de quantification).

---

## WB-LF (E1)

`src/witchesbrew/attacks/lowfreq.py`. `subspace_project` projette alternativement (Dykstra, quelques
balayages) sur `S` (passe-bas DCT en espace pixel) puis sur `{‖δ‖∞ ≤ ε}` ∩ plage pixel valide
(via `PoisonBounds.project` existant). Branché dans `Brewer` quand `delta_param == "lowfreq"` : init
dans le sous-espace + projection à chaque pas. Le reste de l'Algorithme 1 ne bouge pas.

## WB-Ψ (E2)

Brewing conscient de la défense. Deux flags distincts :

- `brew_purify ∈ {none,jpeg,dct,gaussian}` : le Ψ **de brassage**. Quand non nul, le gradient poison
  est pris sur `Ψ(xᵢ+δᵢ)` et moyenné sur `brew_purify_eot` tirages de Ψ — l'espérance
  `E_ξ[Σᵢ ∇ℓ(Ψ_ξ(xᵢ+δᵢ))]` est **à l'intérieur** du cosinus (`gradient_matching_loss(purifier, eot)`).
  JPEG utilise la variante BPDA straight-through, donc `dB/dδ` traverse Ψ.
- `purify ∈ {none,jpeg,dct,gaussian}` : la défense **côté victime**, appliquée aux images empoisonnées
  à l'ingestion dans `train_model` (ordre PureGen : inject → Ψ → augmentation). Le surrogate θ reste
  propre (`purify` forcé à `none` au pré-entraînement).

Les deux sont séparés exprès : brasser contre un Ψ et évaluer sous un autre = test de transfert ;
même Ψ des deux côtés = l'attaque adaptative. Défauts `none` → chemin du papier inchangé.

**Ordre vs augmentation** : au brassage, Ψ s'applique à l'entrée du modèle (après l'augmentation
différentiable si active) ; côté victime, Ψ purifie l'image stockée avant l'augmentation. Pour
reproduire exactement le réglage certifié de Hong et al. (sans augmentation d'entraînement), mettre
`victim_augment=false`.

```bash
# WB-Psi : brasser à travers JPEG, évaluer sous la défense JPEG
python scripts/run_experiment.py --config cifar10/baseline.yaml \
       --set name=wb_psi --set brew_purify=jpeg --set brew_purify_eot=2 --set purify=jpeg
# comparer : WB plein brassé sans conscience de Psi, mais évalué sous la même défense
python scripts/run_experiment.py --config cifar10/baseline.yaml \
       --set name=wb_full_vs_jpeg --set purify=jpeg
```

## Compromis (E3)

`scripts/tradeoff.py` balaie une grille `(eps, lowfreq_cutoff)`, brasse + évalue une expérience par
point, et lit en retour trois quantités par point (moyennées sur les cas, calculées sur le `δ` brassé
par `analysis.spectrum.delta_report`, stockées dans chaque `caseNN.json` sous `delta_metrics`) :

| Quantité | Mesure | Proxy de |
|---|---|---|
| succès | `avg_poison_success` des victimes | efficacité |
| `offmanifold_hf` | fraction d'énergie DCT de `δ` hors du bloc `cutoff×cutoff` | détectabilité EBM (synthétique) |
| `ssim` / `visibility = 1 − ssim` | SSIM global clean vs empoisonné | visibilité |

```bash
python scripts/tradeoff.py --config cifar10/baseline.yaml \
       --eps 16,8 --cutoff 8,12,16 --include-full \
       --set purify=jpeg --set brew_purify=jpeg
python scripts/plot.py tradeoff --file results/<name>/tradeoff.json
```

La courbe elle-même est la contribution (cf. *Pick your Poison*) : en resserrant la coupure, `δ`
rentre dans le sous-espace conservé, la détectabilité et le budget détruit par Ψ chutent, la
visibilité monte, le succès arbitre. `--include-full` ajoute le point WB plein (`delta_param=full`)
comme référence à droite de l'axe des coupures.

---

## Lancer

Le diagnostic tourne sur des poisons **déjà brossés et exportés** (découplage GPU/CPU) :

```bash
# 1. brasser + exporter (GPU) -- WB plein puis WB-LF
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export packed \
       --set name=wb_full
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export packed \
       --set name=wb_lf --set delta_param=lowfreq --set lowfreq_cutoff=12

# 2. diagnostiquer (CPU)
python scripts/diagnose_frequency.py --config cifar10/baseline.yaml \
       --set name=wb_full --poisons poisons/wb_full_case00
python scripts/diagnose_frequency.py --config cifar10/baseline.yaml \
       --set name=wb_lf --set delta_param=lowfreq --poisons poisons/wb_lf_case00

# 3. figure : spectre + survie, WB vs WB-LF
python scripts/plot.py frequency --file results/wb_full/frequency_probe.json \
       --lowfreq results/wb_lf/frequency_probe.json
```

Paramètres : `lowfreq_cutoff` (n ∈ {8,12,16} sur 32), `purify_cutoff`, `jpeg_quality`,
`purify_sigma`, `purify_eot`, `subspace_project_iters`.

### Ce qu'on attend

Le diagnostic sur un vrai lot (GPU, params papier) doit montrer :
- WB plein : énergie concentrée en haute fréquence, `budget_survival` faible sous `dct`/`jpeg`,
  `B_after ≫ B_before` (l'alignement s'effondre après Ψ) ;
- WB-LF : énergie entièrement basse fréquence, `budget_survival ≈ 1` sous `dct`, `B_after ≈ B_before`.

Un smoke sur ImageFolder jouet 16px (brewing bridé) donne déjà la structure : WB plein ~62 %
d'énergie haute fréquence et 6 % de survie sous passe-bas ; WB-LF 0 % haute fréquence et 100 % de
survie. Le contraste E0→E1 est le résultat présentable.

Même smoke bout-en-bout pour E2→E3 (toy, 3 pas, 1 victime), défense JPEG côté victime : WB plein meurt
(succès 0 %, `offmanifold_hf` 0,79), WB-LF survit (succès 50 %, `offmanifold_hf` 0,0). Direction
correcte ; les vrais chiffres demandent le GPU aux params papier.

---

## Pièges (à reporter)

- **Visibilité** : à `ℓ∞` égal, une perturbation basse fréquence est plus visible (Sharma et al.
  2019). Toujours reporter LPIPS/SSIM et des exemples visuels à côté du succès.
- **Gradients obfusqués** (Athalye et al. 2018) : utiliser EOT (`purify_eot > 1`) et vérifier que
  l'attaque reste efficace quand la graine de Ψ change.
- **Coût de la backprop à travers un débruiteur** : rester à un pas (c'est le setting certifié) ;
  la variante BPDA de `jpeg` est là pour le brewing conscient de Ψ.

## Extensions (hooks restants)

- **Vrai DDPM** — remplacer le débruiteur gaussien synthétique par le vrai DDPM CIFAR-10 de Ho et al.
  (setting exact de Hong et al. 2024), branché via `_psi_by_kind`. Le reste d'E2 (BPDA/EOT, hook
  gradient de `Brewer`, défense côté victime) est déjà en place.
- **WB-énergie** — pénalité `λ·G(xᵢ+δᵢ)` avec `G` l'énergie d'un EBM public, pour rester dans la
  variété (PureGen détecte les poisons par leur énergie élevée). Le proxy `offmanifold_hf` d'E3
  anticipe ce signal sans modèle externe.
- **LPIPS** — `delta_report` reporte le SSIM (sans poids externes) ; ajouter LPIPS demande un réseau
  pré-entraîné.
