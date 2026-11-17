# Piste 2 — Witches' Brew contre Deep Partition Aggregation

Extension de recherche, **hors article**. Tout est derrière `dpa_k: 0` (désactivé) : le chemin du
papier ne lit aucun de ces champs. Énoncés et preuves dans `docs/theory/theorie_pistes.tex`, §4.

## La question

DPA (Levine & Feizi, 2021) hache chaque image d'entraînement vers une des `k` partitions, entraîne un
modèle par partition et fait voter. Son certificat ne tolère que `s < g/2` poisons (`g` = marge de
votes), soit une vingtaine à `k = 50`. Pourtant Wang & Feizi (2023) mesurent une robustesse empirique
bien au-delà, et l'expliquent par le *poisoning overfitting* : à fraction fixée, moins d'images par
partition ⇒ l'attaque marche moins. Deux hypothèses sont implicites :

- **H1** : les poisons se répartissent uniformément entre les partitions (le hash les disperse) ;
- **H2** : l'attaque souffre d'overfitting quand le train set rétrécit.

Un attaquant par gradient matching peut violer les deux. H1 parce que le hash de DPA est une fonction
**publique de l'image stockée** : l'attaquant choisit la partition de chaque poison (lemme de
placement). H2 est une question empirique, jamais mesurée pour WB (E0).

## Ce que dit la théorie (§4 du .tex)

| Énoncé | Contenu | Testé par |
|---|---|---|
| Certificat (Thm DPA) | `s < g/2` ⇒ prédiction inchangée ; `m_0 = ⌊g/2⌋ + 1` modèles à retourner | `clean.margin`, `clean.certified_radius`, `clean.m0` |
| Effet Condorcet | sous H1 sans overfitting, succès agrégé = `P[Bin(k, p) ≥ m_0]` : amplifie `p < 1/2` vers 0 et `p > 1/2` vers 1 | `dpa_uniform` |
| Lemme de placement | somme de pixels : `+1` sur `t < k` pixels ; hash pondéré public : recherche ±1, `≈ k` essais ; hash secret : impossible | `placement.mean_shift_pixels`, `placement.mean_trials`, `dpa_hash=secret` lève |
| Thm d'allocation | il faut `m ≥ m_0` partitions ; répartition égale si `p(ρ)` concave ; `S(m) = P[Bin(m, p(Bk/(mn))) ≥ m_0]` | balayage de `dpa_attack_partitions` |
| Budget critique | gain `k/m_0 ≈ 2k/g` sur le budget nécessaire par rapport à H1 | `dpa_conc*` contre `dpa_uniform` à budget égal |

## Ce qui est implémenté

### Défenseur — `src/witchesbrew/defenses/dpa.py`

- `partition_hash(images_uint8, k, mode, key)` : trois familles.
  - `sum` : somme des pixels 8 bits mod `k` (implémentation d'origine de DPA).
  - `weighted` : `Σ W_j x_j mod (2^31 − 1) mod k`, `W` tirée de `dpa_hash_key` (**publique**, Wang & Feizi Déf. 4.1).
  - `secret` : même formule, `W` tirée de `dpa_secret_key`, inconnue de l'attaquant.
- `partition_dataset(...)` : partition de chaque index ; les poisons sont hachés **après** ajout de δ et quantification 8 bits, car c'est l'image stockée que le défenseur voit.
- `train_dpa(...)` : `k` modèles de base avec la recette victime, chacun sur `Subset(trainset, partition)` ; le modèle `i` est semé `seed + i`, à l'identique pour la référence propre et le run empoisonné (comparaison appariée). Vote, marge `g`, rayon certifié, `m_0`, précision de validation de l'agrégat.

### Attaquant — `src/witchesbrew/attacks/placement.py`

- `choose_partitions(k, m, pick, ...)` : `random`, ou `weakest` = les `m` partitions dont le modèle propre tient la cible avec la plus petite marge de logits (`logit[pred] − logit[y_adv]`).
- `allocate(poison_ids, partitions)` : répartition égale `B/m` (Thm d'allocation (ii), régime concave).
- `place_in_partition(...)` : édite δ **après** brassage pour que `hash(quantize(x + δ)) ≡ partition visée`. Coût `1/255` en ℓ∞. Le brassage se fait à `eps − dpa_placement_margin` (défaut 2 : 1 pour le décalage, 1 pour l'arrondi 8 bits), le δ final reste dans la boule `eps` (vérifié par assertion). Le δ renvoyé reproduit exactement l'image 8 bits dont le hash a été vérifié.
- Second levier (`dpa_partition_surrogate=true`) : chaque groupe de poisons est brassé contre un surrogate entraîné sur **sa** partition propre (≈ 950 images sur Imagenette), plus fidèle au modèle de base que la victime entraînera.

### Pipeline — `evaluate.run_dpa_case`, `scripts/run_dpa.py`

1. Référence propre : `k` modèles propres sur les partitions naturelles → votes, `g`, `m_0`, marges par partition. Mise en cache (`checkpoints/dpa_clean_*.json`) : toutes les cellules d'un même cas la partagent.
2. Allocation : `m = dpa_attack_partitions` partitions (0 = H1, pas de placement).
3. Brassage à `eps − marge` (surrogate global, ou par partition).
4. Placement, puis vérification que 100 % des poisons sont dans leur partition.
5. Victime DPA sur le jeu stocké → vote ; succès = l'agrégat prédit `y_adv`.
6. `dpa_single_victim=true` : entraîne aussi la victime unique (sans DPA) sur le même δ, pour la comparaison « avec / sans défense à budget égal ».

Par partition contenant des poisons, le JSON enregistre `rho = poisons / taille`, `flipped` (le modèle de base prédit `y_adv`), la marge propre et `L`. **`flip_rate` est la courbe `p(ρ)` du théorème d'allocation, mesurée** : un balayage de `m` donne `ρ = B/(m·n/k)` à différents niveaux.

## Protocole (Imagenette, k = 10, budget 5 %)

```bash
BASE="--config piste2/dpa_imagenette.yaml --set workers=8 --set progress=true"
# H1 : hash naturel, pas de placement (le réglage que Wang & Feizi supposent)
python scripts/run_dpa.py $BASE --set name=dpa_uniform --set dpa_single_victim=true
# E2 : concentré sur m partitions ; m_0 se lit dans clean.m0 (vote unanime : m_0 = 6 à k = 10)
python scripts/run_dpa.py $BASE --set dpa_attack_partitions=3 --set dpa_pick=weakest --set name=dpa_conc3
python scripts/run_dpa.py $BASE --set dpa_attack_partitions=6 --set dpa_pick=weakest --set name=dpa_conc6
# second levier : surrogate par partition
python scripts/run_dpa.py $BASE --set dpa_attack_partitions=6 --set dpa_pick=weakest --set dpa_partition_surrogate=true --set name=dpa_conc6_ps
# hash pondéré public : le placement marche encore (lemme (b))
python scripts/run_dpa.py $BASE --set dpa_hash=weighted --set dpa_hash_key=1 --set dpa_attack_partitions=6 --set name=dpa_conc6_weighted
# E3 : hash secret, placement impossible -> doit ressembler à dpa_uniform
python scripts/run_dpa.py $BASE --set dpa_hash=secret --set dpa_secret_key=42 --set name=dpa_secret
# E0 : poisoning overfitting pour WB (Table 7 réutilisée, fraction de poison constante)
python scripts/sweep.py --config piste2/overfitting.yaml --grid data_ablation=1,0.5,0.2,0.1
```

Coût par cas : `k` modèles propres (une fois, cache) + `k` modèles victimes, chacun sur `1/k` du train
set ⇒ ≈ 2 entraînements complets + brassage. Avec `dpa_single_victim`, un entraînement de plus.

## Lecture

- `dpa_uniform` : la référence H1. Avec `k = 10` et 474 poisons, chaque partition reçoit ≈ 47 poisons sur ≈ 950 images (5 %) : si `p(5 %)` sur 950 images est petit (overfitting, H2), peu de modèles basculent et l'agrégat tient.
- `dpa_conc3` (`m < m_0`) : ne **peut pas** faire basculer un vote unanime (Thm (i)) ; sert à mesurer `p(ρ)` à `ρ ≈ 16 %`.
- `dpa_conc6` (`m = m_0` pour un vote unanime) : le test du théorème. Succès agrégé attendu `≈ p(8 %)^6` avec `flip_rate ≈ p(8 %)`.
- Comparer `flip_rate` entre `uniform` (ρ = 5 %) et `conc6` (ρ ≈ 8 %) et `conc3` (ρ ≈ 16 %) donne trois points de `p(ρ)`.
- `dpa_secret` ≈ `dpa_uniform` : la robustesse empirique de DPA repose sur le secret du hash, pas sur l'agrégation.
- Sur un vote **non unanime** (petite marge `g` dans `clean.margin`), `m_0` est petit et le gain `2k/g` est grand : ce sont les cibles à regarder en premier dans `per_partition`.

## Limites

- Le régime concave de `p(ρ)` est supposé pour la répartition égale ; si la courbe mesurée est convexe à bas `ρ`, concentrer sur exactement `m_0` partitions est le choix du théorème (c'est `dpa_conc6` pour un vote unanime).
- `k = 10` (Imagenette) et non `k = 50` (CIFAR-10 des articles) : partitions de 950 images, pas de 1 000, et un vote sur 10 modèles seulement, donc une marge entière grossière.
- Les modèles de base partagent la recette victime (40 époques sur 950 images) : ils sur-apprennent plus que sur le train set complet, ce qui est précisément H2 et fait partie de ce qu'on mesure.
