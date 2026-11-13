# Guide : les deux bases de code, et à quoi chacune sert

Ce document répond à quatre questions : à quoi sert le dépôt de l'auteur, à quoi sert celui-ci, ce
que « valider le papier » veut dire concrètement, et dans quel ordre lancer les choses.

Il ne suppose aucune connaissance des deux dépôts. Les autres documents de `docs/` sont plus
techniques : lisez celui-ci d'abord.

---

## 0. Le modèle mental, avant toute commande

La confusion vient presque toujours du fait que l'expérience met en jeu **deux réseaux différents**
et **deux métriques différentes**. Tant que ces quatre objets ne sont pas distincts dans votre tête,
aucune sortie de terminal n'a de sens.

### Les deux réseaux

| | Nom dans le papier | Rôle | Qui le voit |
|---|---|---|---|
| **Réseau 1** | le modèle « pré-entraîné », `theta` (§3.3) | Entraîné sur des données **propres**. L'attaquant s'en sert pour calculer les gradients. Il est **gelé** pendant l'attaque. | L'attaquant seulement |
| **Réseau 2** | la **victime** | Fraîchement initialisée, entraînée **à partir de zéro** sur le jeu de données empoisonné. | La victime seulement |

Le réseau 1 ne sert qu'à fabriquer la perturbation. Il est ensuite jeté. Le succès ou l'échec se
mesure uniquement sur le réseau 2, et sur **huit** réseaux 2 différents, chacun avec une
initialisation aléatoire nouvelle (annexe B). C'est tout l'intérêt : l'attaque doit survivre à des
modèles qu'elle n'a jamais vus.

Dans ce dépôt, le réseau 1 s'appelle le *surrogate* et est mis en cache dans `checkpoints/`. Dans le
dépôt de l'auteur, c'est le même objet Python `Victim` qui est d'abord entraîné propre, puis réutilisé
pour le *brewing*, puis réinitialisé pour chaque victime.

### Les deux métriques

| Métrique | Ce qu'elle mesure | Valeur attendue |
|---|---|---|
| **`avg. poison success`** | Sur **une seule image de test prédéterminée** : la victime la classe-t-elle dans la classe choisie par l'attaquant ? Moyenne sur 10 cas × 8 victimes. | **91,25 %** (±6,14) pour la ligne de référence |
| **`clean validation acc`** | Précision globale de la victime sur le jeu de validation, **image cible exclue** (annexe F.9). | **92,20 %**, soit ~0,1 point de moins qu'un modèle propre |

**La précision globale ne s'effondre pas, et ce n'est pas un bug.** C'est le cœur de la menace.
L'attaque est une attaque d'*intégrité*, pas de *disponibilité* : le modèle reste excellent, donc rien
dans les métriques habituelles n'alerte, mais une image précise est classée comme l'attaquant le
veut. Si vous cherchiez une chute d'accuracy, vous cherchiez la mauvaise chose.

### Ce qu'est un « cas »

Un *cas* (`case`) est un scénario de poisonnement : une classe cible, une classe adverse, et l'ID
d'une image cible dans le jeu de test. Exemple, cas 0 de la table 4 : la cible est un `dog` de l'ensemble
de test (ID 8745), que l'attaquant veut faire classer `frog`, et les 500 images perturbées sont
des `frog` du jeu d'entraînement, **étiquetées `frog`, ce qui est leur vraie étiquette**. Aucune
étiquette n'est modifiée : c'est ce que « clean-label » veut dire.

Le papier publie 10 cas (table 4 pour CIFAR-10, table 5 pour ImageNet). Un seul cas ne donne pas de
chiffre exploitable : avec une seule cible et une seule victime, le résultat est binaire, 0 % ou
100 %. Les 10 cas × 8 victimes existent précisément pour cette raison.

---

## 1. Le dépôt de l'auteur : `poisoning-gradient-matching/`

### Ce que c'est

Le code qui a produit les chiffres de l'article, écrit par Jonas Geiping et Liam Fowl. C'est la
**référence normative** : quand le papier est muet sur un détail, ce dépôt donne la réponse que les
auteurs ont effectivement utilisée.

C'est un code de recherche : un script en ligne de commande avec environ 60 drapeaux, et sept
fichiers `.sh` qui encodent les expériences du papier. Il ne calcule **pas** les moyennes qu'il
publie — c'est à vous de le faire.

### Installation

Aucun paquet à installer, aucun `setup.py`. Il suffit de PyTorch ≥ 1.6 et torchvision, et de lancer
le script depuis la racine du dépôt.

```bash
cd poisoning-gradient-matching
python brew_poison.py --help          # la liste complète des drapeaux
```

### La structure en trois classes

| Classe | Fichier | Rôle |
|---|---|---|
| `Kettle` | `forest/data/kettle.py` | Les données : choisit la cible, choisit les 500 images à empoisonner, construit les `DataLoader`, garde la table `poison_lookup` (ID d'image → ligne dans le tenseur de perturbation) |
| `Victim` | `forest/victims/victim_single.py` | Le réseau : l'entraîne, le réinitialise, calcule des gradients |
| `Witch` | `forest/witchcoven/witch_matching.py` | L'attaque : prend un `Victim` et un `Kettle`, renvoie `poison_delta` |

`poison_delta` est un tenseur qui contient **seulement les perturbations**, pas les images. Les
perturbations sont ajoutées à leurs images au moment de former chaque mini-lot.

### Une exécution complète

```bash
python brew_poison.py --net ResNet18 --vruns 8 --poisonkey 2000000000 \
                      --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --name mon_test
```

Ce qui se passe, dans l'ordre (`brew_poison.py`) :

1. `model.train(data)` — entraîne le réseau 1 sur des données propres, 40 époques.
2. `witch.brew(model, data)` — l'algorithme 1 : 8 restarts × 250 pas de *signed Adam*. Renvoie `poison_delta`.
3. `model.validate(data, poison_delta)` — réinitialise le réseau 8 fois (`--vruns 8`) et l'entraîne
   chaque fois 40 époques sur les données empoisonnées, en vérifiant la cible à la fin.
4. `record_results(...)` — ajoute **une ligne** à `tables/ResNet18_single-class_mon_test.csv`.

Coût : environ 1,5 à 3 heures sur un GPU moderne, pour **un seul cas**.

### Drapeaux qu'il faut connaître

| Drapeau | Effet |
|---|---|
| `--poisonkey` | La graine du cas. Les dix valeurs du papier sont dans `defaults.sh` : `2000000000`, `2100000000`, … `2111111111` |
| `--vruns` | Nombre de victimes réentraînées. Le papier utilise 8 |
| `--net` | Architecture. **`ConvNet64`, pas `ConvNet`**, pour la table 2 : `ConvNet` est en largeur 32, le papier décrit la largeur 64 |
| `--recipe` | `gradient-matching`, `gradient-matching-private`, `poison-frogs`, `watermark`, `metapoison` |
| `--eps`, `--budget` | La force de l'attaque. Défauts : 16 et 0,01 |
| `--restarts`, `--attackiter`, `--tau` | `R`, `M`, et le pas. Défauts : 8, 250, 0,1 |
| `--ensemble` | `K`. Défaut 1 |
| `--paugment` | **Attention, logique inversée** : `action='store_false'`. Mettre le drapeau **désactive** l'augmentation différentiable |
| `--noaugment` | Désactive l'augmentation de la victime |
| `--vnet` | Évalue sur une autre architecture que celle du brewing (expériences de transfert) |
| `--save full` | Exporte les images empoisonnées en PNG dans `poisons/` |
| `--dryrun` | Un seul lot, un seul pas, une seule époque. Pour vérifier que rien ne plante |

Tous les drapeaux sont dans `forest/options.py`.

### Comment obtenir le 91,25 % du papier avec ce dépôt

C'est entièrement manuel :

```bash
bash defaults.sh     # 10 processus, un par poisonkey, ~15 à 30 heures de GPU
```

Puis vous ouvrez `tables/ResNet18_single-class_def.csv`, qui contient maintenant 10 lignes, et vous
faites vous-même la moyenne de la colonne `target_acc_reinit` (chaque ligne est déjà la moyenne sur
les 8 victimes de ce cas) et son erreur standard. Ce chiffre-là est le 91,25 % ± 6,14.

Il n'y a **pas de reprise après interruption** : si le processus du cas 7 meurt, ce cas est à refaire
entièrement. Il n'y a pas non plus de comparaison automatique aux chiffres publiés.

### Ce dépôt sert à quoi, pour vous

Trois choses, et pas une quatrième :

1. **Trancher les ambiguïtés du papier.** Le papier dit « decaying step size » sans donner les
   jalons ; `witch_base.py` donne `M/2.667, M/1.6, M/1.142`. C'est ainsi que `docs/DESIGN_DECISIONS.md`
   a été rempli, et chaque emprunt y porte le tag `[C]`.
2. **Être l'arbitre en cas de désaccord numérique.** Si mon code donne 75 % là où le papier dit
   91 %, la question « ai-je mal lu une équation ? » se règle en comparant à ce code-là.
3. **Les expériences que je n'ai pas réimplémentées** : le brewing distribué multi-GPU, MetaPoison,
   le *watermarking*, TinyImageNet, l'export AutoML, l'interface avec le *benchmark* de
   Schwarzschild et al.

Ce n'est **pas** la bonne base pour mener une campagne d'expériences : pas de reprise, pas
d'agrégation, pas de tests, et quelques bugs réels listés en §4 ci-dessous.

---

## 2. Ce dépôt : `witches_brew/`

### Ce que c'est

Une réimplémentation écrite depuis le papier, qui automatise le **protocole** en plus de l'attaque.
Le papier est dans `docs/paper/witchesbrew.pdf`. Chaque valeur d'hyperparamètre est soit tirée du
papier, soit tirée du dépôt de l'auteur et signalée comme telle dans `docs/DESIGN_DECISIONS.md`.

La différence tient en une phrase : le dépôt de l'auteur vous donne **un cas par invocation et une
ligne de CSV** ; celui-ci vous donne **le chiffre publié, avec son erreur standard, et la
comparaison au papier**.

### Installation

```bash
cd witches_brew
pip install -e .
pytest tests/          # 112 tests, CPU, ~15 s
```

Les 112 tests ne sont pas décoratifs : ils épinglent l'implémentation aux équations. `B` est comparé
à un cosinus reconstruit indépendamment en float64 (accord à 9e-13), `dB/dΔ` à des différences
finies centrées (erreur relative 1,3e-07), et `ε` est vérifié en unités pixel après dénormalisation
et quantification 8 bits (exactement 16/255). Si `pytest` passe, la méthode est juste ; ce qui reste à
vérifier, c'est si elle **reproduit les chiffres**, et ça demande un GPU.

### Les quatre scripts

```bash
# 1. Une expérience complète : 10 cas, 8 victimes chacun, puis l'agrégat
python scripts/run_experiment.py --config cifar10/baseline.yaml

# 2. Une grille (les figures de balayage du papier)
python scripts/sweep.py --config cifar10/inset_RK_sweep.yaml --grid restarts=1,8 ensemble=1,2,4,8

# 3. Les deux défenses de la section 5.3
python scripts/run_defense.py dp --config defenses/dp_sweep.yaml
python scripts/run_defense.py filtering --config defenses/filtering.yaml

# 4. Fabriquer les poisons d'un cas et les regarder, sans entraîner de victime
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export png
```

Puis, à tout moment :

```bash
python scripts/aggregate.py          # tout ce qui existe, à côté des chiffres publiés
python scripts/aggregate.py --table table2
python scripts/plot.py sweep --file results/inset_RK_sweep_sweep.json --x ensemble --series restarts
```

### La configuration

Pas de drapeaux. Un fichier YAML par résultat publié, 29 au total, empilés sur `configs/base.yaml`
qui contient les valeurs du papier. Les tests (`tests/test_configs.py`) vérifient que chaque fichier
nommé d'après un résultat porte bien le réglage de ce résultat — par exemple que
`table2_convnet_gm.yaml` utilise `eps: 32`, comme la légende de la table 2 l'exige.

Pour changer une valeur depuis la ligne de commande :

```bash
python scripts/run_experiment.py --config cifar10/baseline.yaml --set victim_runs=2 --set case_ids=0
```

Tout `--set` est enregistré dans le JSON de résultat et met `paper_setting: false`. Un run bridé ne
peut donc pas être confondu, trois semaines plus tard, avec une reproduction du protocole publié.

### Les paramètres qui comptent

| Champ | Défaut | Ce qu'il contrôle |
|---|---|---|
| `budget` | `0.01` | Fraction du jeu d'entraînement perturbée. 1 % = 500 images sur CIFAR-10 |
| `eps` | `16.0` | Borne `l_inf`, en unités pixel 0–255 |
| `targets` | `1` | Nombre d'images cibles (`T` de l'eq. 1) |
| `tau` | `0.1` | Pas du *signed Adam* de l'attaque |
| `restarts` | `8` | `R`, nombre de redémarrages. Le meilleur `B` gagne |
| `attack_steps` | `250` | `M`, pas d'optimisation par restart |
| `ensemble` | `1` | `K`, nombre de réseaux 1. Le papier montre que `K=1` suffit grâce à l'augmentation différentiable |
| `differentiable_augment` | `true` | L'attaque anticipe-t-elle l'augmentation de la victime ? C'est l'ingrédient clé de la table 1 |
| `victim_augment` | `true` | La victime augmente-t-elle ses données ? |
| `net` / `victim_net` | `ResNet18` | Architecture du brewing / de la victime. Les séparer donne les expériences de transfert |
| `epochs`, `lr`, `lr_milestones` | `40`, `0.1`, `[14,24,35]` | La recette victime de l'annexe B.2 |
| `cases`, `victim_runs` | `10`, `8` | Le protocole de l'annexe B |
| `recipe` | `gradient-matching` | Ou `poison-frogs`, ou `bullseye`, pour la table 2 |
| `objective` | `cosine` | L'eq. 3. `euclidean` optimise l'eq. 2 directement (table 8, figure 12) |
| `data_ablation` | `1.0` | Fraction des données gardée par la victime (table 7) |

### Les fichiers du code, par ordre d'importance

| Fichier | Contenu |
|---|---|
| `src/witchesbrew/attacks/objectives.py` | `B` lui-même. `create_graph=True` est tout le truc : le gradient-paramètres reste dérivable par rapport aux pixels |
| `src/witchesbrew/attacks/brew.py` | L'algorithme 1 : restarts, *signed Adam*, projection sur la boule `ε` ∩ plage pixel valide |
| `src/witchesbrew/data/augment.py` | L'augmentation différentiable (`grid_sample`), partagée par l'attaque et la victime |
| `src/witchesbrew/data/setup.py` | Construit un cas à partir des tables 4 et 5 publiées |
| `src/witchesbrew/train.py` | La recette de l'annexe B.2, l'injection des perturbations par index, la défense DP |
| `src/witchesbrew/evaluate.py` | Le protocole de l'annexe B : surrogate → brewing → 8 victimes → moyenne ± erreur standard |
| `src/witchesbrew/diagnostics.py` | Les figures 2, 5, 9, 11, que le dépôt de l'auteur ne produit pas |
| `src/witchesbrew/defenses/` | DP (figure 4b) et filtrage d'anomalies (figure 4a, table 6) |

### Où sortent les résultats

```
results/<nom>/case00.json      un fichier par cas, avec la config complète qui l'a produit
results/<nom>/case01.json
...
checkpoints/                   les surrogates, mis en cache par (architecture, largeur, époques, graine)
poisons/                       les perturbations exportées, si save_poisons est mis
results/figures/               les sorties de plot.py
```

Comme chaque cas est écrit dès qu'il est fini et que le surrogate est en cache, **une exécution
interrompue reprend où elle s'est arrêtée**. Relancez exactement la même commande.

---

## 3. En quoi tout ceci « valide » le papier

C'est la question la plus importante, et elle a trois niveaux distincts. Les confondre est la source
principale de frustration.

### Niveau 1 — Validité mathématique : la formule est-elle la bonne ?

Se vérifie sur CPU, en quinze secondes, sans aucune donnée réelle.

```bash
pytest tests/
```

Ce que ça établit : `B` est bien l'équation 3 ; son gradient par rapport à `Δ` est correct ; la
perturbation ne sort jamais de la boule `ε`, y compris après quantification en 8 bits ; les poisons
viennent bien de la classe adverse avec leur vraie étiquette ; la recette victime est bien celle de
l'annexe B.2 ; les lignes de « hasard pur » de la table 6 se recalculent exactement (200/300 et
400/600). Le détail est dans `docs/AUDIT.md`.

Ce que ça n'établit pas : que l'attaque **marche**.

### Niveau 2 — Validité de l'implémentation : le pipeline tourne-t-il ?

```bash
python scripts/run_experiment.py --config smoke_cpu.yaml
```

Environ une minute sur CPU. Ce que ça établit : les données se chargent, le cas se construit, le
brewing converge, la victime s'entraîne, le JSON s'écrit.

Ce que ça n'établit pas : quoi que ce soit de numérique. Cette configuration brise délibérément tous
les paramètres — `M=5` au lieu de 250, `R=1` au lieu de 8, 1 époque au lieu de 40, largeur 16 au lieu
de 64, 6 % des données. **`0.00 %` de succès et `15.80 %` de précision sont la sortie attendue.** La
table 8 du papier montre que passer `M` de 250 à 50 fait déjà tomber le succès de 91,25 % à 40 % ;
`M=5` ne peut rien donner. Si vous avez pris ce `0.00 %` pour un échec, c'était le malentendu.

### Niveau 3 — Validité scientifique : les chiffres publiés sortent-ils ?

C'est le seul niveau qui exige un GPU et des heures. Reproduire le papier, concrètement, c'est
retrouver ces chiffres-là :

| Réglage | Papier |
|---|---|
| Référence : ResNet-18, 1 %, `eps=16`, `K=1`, `R=8`, `M=250` | **91,25 %** (±6,14) de succès, 92,20 % de précision propre |
| Sans augmentation différentiable, victime qui augmente | 32,50 % (±12,27) |
| Objectif Poison Frogs, même cadre | 1,25 % |
| Objectif Bullseye, même cadre | 3,75 % |
| `M=50` au lieu de 250 | 40,00 % |
| Filtrage d'anomalies à 10 % (budget 4 %) | 248 poisons / 252 propres retirés, contre 200 / 300 au hasard |

Et c'est là que la *structure* des expériences devient le vrai contenu scientifique. Prenez la
table 1 : les quatre lignes ne servent pas à montrer que l'attaque marche, elles servent à établir
**pourquoi** elle marche. Sans aucune augmentation nulle part, le succès est de 100 % — facile mais
irréaliste. Dès que la victime augmente ses données et que l'attaquant l'ignore, on tombe à 32,50 % :
l'attaque est détruite par un simple recadrage aléatoire. Un ensemble de 8 réseaux récupère 78,75 %
pour huit fois le coût. L'augmentation différentiable, avec **un seul** réseau et donc le coût de
départ, donne 91,25 %. La conclusion de la table est là : l'augmentation différentiable remplace un
ensemble de 8 modèles sans coût supplémentaire.

De même, la table 2 échange **seulement l'objectif** et garde tout le reste fixe — même optimiseur,
mêmes restarts, même augmentation, même évaluation. Les 1,25 % de Poison Frogs contre les 90 % de
l'appariement de gradients ne mesurent donc pas deux implémentations différentes : ils mesurent deux
équations, dans un cadre identique. C'est pour cette raison que `objectives.py` contient les quatre
objectifs côte à côte et que le reste du pipeline ne change pas d'une ligne.

C'est ce que « valider » veut dire ici : non pas « le code tourne », mais « ces écarts-là
réapparaissent, à l'erreur standard près, sur du matériel différent et une implémentation
indépendante ».

### Ce que la reproduction ne peut pas établir

Quatre choses sont hors d'atteinte, et `docs/DEVIATIONS.md` les détaille :

- **Les ID exacts des images empoisonnées.** Le papier publie les graines et les ID de cible, jamais
  les 500 ID de poison. Le dépôt de l'auteur les tire d'un générateur aléatoire seedé ; je les tire
  avec NumPy. Même cible, sous-ensemble différent. C'est précisément pourquoi le papier moyenne sur
  10 cas.
- **La table 3 à l'identique.** Les 100 scénarios figés et le code d'évaluation vivent dans le dépôt
  de Schwarzschild et al., pas dans ce papier.
- **AutoML Cloud (§5.2).** Service payant de Google.
- **ImageNet depuis zéro.** Les auteurs ont utilisé 4 GPU. Le *brewing* ImageNet est bon marché
  (« moins de quatre heures de GPU ») parce qu'il part de poids pré-entraînés ; ce sont les
  entraînements victimes qui coûtent.

---

## 4. Pourquoi ne pas simplement utiliser le dépôt de l'auteur

Trois raisons, par ordre d'importance.

### 4.1 Des bugs réels, dont deux touchent des résultats publiés

Trouvés en lisant le code ligne à ligne, corrigés ici :

| Où | Problème | Effet |
|---|---|---|
| `witch_matching.py:138` | `grad.mul(clip_coef)` au lieu de `mul_`. Non in-place, résultat jeté | L'écrêtage de gradient de l'« attaquant averti » (annexe F.3) ne fait rien. Seul le bruit s'applique. Touche la figure 4b |
| `witch_base.py:175,190` | Le score de restart est la perte accumulée **pendant** la dernière passe, donc celle du `Δ` d'**avant** le dernier pas et la projection | L'algorithme 1 ligne 10 est censé sélectionner le `Δ` retourné. Avec `R=8`, le mauvais jeu de poisons peut être retenu |
| `witch_poison_frogs.py:48` | `bypass_last_layer` suppose que le pooling est un module enfant ; sur MobileNetV2 il est dans `penultimate()` | Poison Frogs et Bullseye plantent sur cette architecture |
| `utils.py:47` | `cw_loss` prend le `max` sur **tous** les logits, cible incluse | Perte et gradient nuls dès que la cible est déjà bien classée. Touche la ligne CW de la table 8. Son propre `cw_loss2`, juste en dessous, est la marge correcte |
| `victim_single.py:73` | `single_setup` omet `self.defs`, donc `_step` reçoit les arguments décalés | Chemin `--step` cassé |
| `models.py:29` | `ConvNet128` est identique à `ConvNet64` | Copier-coller |

Il y a aussi un piège de nommage qui n'est pas un bug mais coûte une journée si on le rate :
`--net ConvNet` est en largeur 32, alors que l'annexe B.2 décrit la largeur 64. La table 2 du papier
a bien été produite avec `--net ConvNet64`.

### 4.2 Le protocole n'est pas dans le code

Dix invocations shell séparées, une ligne de CSV chacune, moyenne à faire à la main, aucune reprise
après interruption, aucune comparaison aux chiffres publiés. Ici, `run_experiment.py` **est** le
protocole de l'annexe B, et `aggregate.py` imprime votre colonne à côté de celle du papier avec
l'écart en points.

### 4.3 Ce qui n'y est pas du tout

- Le filtrage par anomalies (figure 4a, table 6) : absent du dépôt de l'auteur.
- Les figures 2, 5, 9, 11 : tracées dans l'article, mais aucun code pour les produire.
- Les deux régimes de la table 7 : `--ablation` réduit le jeu de la victime **et** du surrogate
  ensemble, donc le régime « victim ablation » seul est inatteignable. Ici, `ablate_surrogate` les
  sépare.
- Des tests : trois scripts de vérification manuelle, contre 112 tests ici.

Inversement, le dépôt de l'auteur a des choses que je n'ai pas : brewing distribué multi-GPU,
MetaPoison, watermarking, TinyImageNet, LMDB, export AutoML, interface *benchmark*.

---

## 5. Quoi lancer, dans quel ordre, sur votre machine GPU

Une échelle, du moins cher au plus cher. Ne montez d'un barreau que quand le précédent est propre.

### Barreau 1 — Les tests (CPU, 15 s)

```bash
cd witches_brew
pip install -e .
pytest tests/
```

Attendu : 112 passés. Si oui, la méthode est correcte, indépendamment de tout GPU.

### Barreau 2 — Le smoke test (CPU, ~1 min)

```bash
python scripts/run_experiment.py --config smoke_cpu.yaml
```

Attendu : `0.00 %` de succès, précision autour de 15 %, et un `results/smoke_cpu/case00.json`.
C'est le résultat correct. On vérifie seulement que le pipeline tourne de bout en bout.

### Barreau 3 — Regarder les poisons (GPU, ~30 min)

```bash
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export png
```

Ne fabrique que les poisons, n'entraîne aucune victime. À vérifier dans la sortie JSON :

- `poison_count: 500` (1 % de 50 000)
- `dog -> frog` (cas 0 de la table 4)
- `brew_loss` nettement sous `1.0` — c'est `B`, et `B < 1` signifie un cosinus positif, donc des
  gradients alignés. C'est le signe que l'attaque a convergé
- `quantization_error` qui reste dans la borne après passage en 8 bits

Et regardez les PNG : les figures 6 à 8 du papier affirment que les poisons sont « à peine
distinguables » des originaux. C'est vérifiable à l'œil, gratuitement.

### Barreau 4 — Un cas complet (GPU, ~1,5 à 3 h)

```bash
python scripts/run_experiment.py --config cifar10/baseline.yaml \
  --set case_ids=0 --set victim_runs=2 --set name=check_case0
```

Attendu : la **précision propre doit être autour de 92 %**. C'est la vraie vérification à ce stade.
Le succès de poison, lui, ne vaut rien ici : avec un cas et deux victimes, il ne peut valoir que 0 %,
50 % ou 100 %. Ne concluez rien de sa valeur.

### Barreau 5 — La ligne de référence (GPU, ~15 à 30 h)

```bash
python scripts/run_experiment.py --config cifar10/baseline.yaml
python scripts/aggregate.py --table baseline
```

Attendu : **91,25 % ± 6,14** et 92,20 % de précision. C'est le chiffre phare du papier. L'exécution
est reprenable : si elle s'interrompt au cas 6, relancez la même commande.

### Barreau 6 — Les expériences qui expliquent pourquoi

La ligne de référence seule ne prouve qu'une chose : l'attaque marche. Ce sont les ablations qui
portent le contenu scientifique. Par ordre de rapport intérêt/coût :

```bash
# Table 1 : l'augmentation différentiable remplace un ensemble de 8 modèles. 4 cellules.
for c in table1_K1_noaug_victim_noaug table1_K1_noaug_victim_aug \
         table1_K8_noaug_victim_noaug table1_K1_diffaug_victim_aug; do
  python scripts/run_experiment.py --config cifar10/$c.yaml
done

# Table 2 : l'objectif seul change. 91 % contre 1,25 % contre 3,75 %.
for c in table2_resnet18_gm table2_resnet18_frogs table2_resnet18_bullseye; do
  python scripts/run_experiment.py --config cifar10/$c.yaml
done

# Section 5.3 : les défenses. Le filtrage échoue, la DP coûte très cher.
python scripts/run_defense.py filtering --config defenses/filtering.yaml
python scripts/run_defense.py dp --config defenses/dp_sweep.yaml

python scripts/aggregate.py
```

La table complète des expériences, avec une ligne par résultat publié, son coût et la commande, est
dans `docs/EXPERIMENTS.md`.

---

## 6. Correspondance des deux interfaces

Utile quand vous lisez un `.sh` de `paper_experiments/` et voulez le réglage équivalent ici.

| Dépôt de l'auteur | Ici |
|---|---|
| `--poisonkey 2000000000` | `case_ids: [0]` (les dix graines sont les dix cas des tables 4 et 5) |
| `--vruns 8` | `victim_runs: 8` |
| `--net ResNet18` | `net: ResNet18` |
| `--net ConvNet64` | `net: ConvNet` (largeur 64, celle du papier) |
| `--vnet MobileNetV2` | `victim_net: MobileNetV2` |
| `--eps 16 --budget 0.01` | `eps: 16.0`, `budget: 0.01` |
| `--restarts 8 --attackiter 250 --tau 0.1` | `restarts: 8`, `attack_steps: 250`, `tau: 0.1` |
| `--ensemble 1` | `ensemble: 1` |
| `--paugment` (désactive !) | `differentiable_augment: false` |
| `--noaugment` | `victim_augment: false` |
| `--recipe poison-frogs` | `recipe: poison-frogs` |
| `--recipe gradient-matching-private` | `attacker_aware_dp: true` |
| `--loss similarity` (défaut) | `objective: cosine`, `cosine_mode: global` |
| `--loss cosine1` | `cosine_mode: per_layer` |
| `--loss SE` | `objective: euclidean` |
| `--target_criterion cw` | `target_criterion: carlini-wagner` |
| `--optimization private --gradient_noise 1e-4` | `dp_clip: 1.0`, `dp_noise: 1e-4` |
| `--ablation 0.7` | `data_ablation: 0.7` |
| `--pbatch 512` | `poison_batch_size: 512` |
| `--save full` | `save_poisons: imagefolder` |
| `--dryrun` | `dryrun: true` |

---

## 7. Résumé en six lignes

- Le dépôt de l'auteur est la **référence normative** : il tranche les ambiguïtés du papier et sert
  d'arbitre en cas de désaccord numérique. Il n'est pas commode pour mener une campagne.
- Ce dépôt est l'**instrument de travail** : il automatise le protocole de l'annexe B, reprend après
  interruption, compare aux chiffres publiés, et est couvert par 112 tests.
- `pytest` valide les **équations**. Le smoke test valide le **pipeline**. Seul un GPU valide les
  **chiffres**.
- Un seul cas ne donne jamais un chiffre exploitable : le protocole est 10 cas × 8 victimes, et c'est
  pour cela qu'il coûte cher.
- La précision globale ne s'effondre pas, et c'est tout le point : attaque d'intégrité, pas de
  disponibilité.
- Lisez `docs/DEVIATIONS.md` avant de comparer un seul chiffre au papier.

## 8. Au-delà du papier

Deux extensions hors-article vivent dans le dépôt, derrière des drapeaux à défaut neutre (le chemin
du papier ne bouge pas) :

- **Tes propres données** — `docs/CUSTOM_DATA.md` : mode `dataset: custom`, ImageFolder quelconque,
  cible explicite ou aléatoire.
- **Sonde fréquentielle (piste 1)** — `docs/FREQUENCY_PROBE.md` : où WB dépense son budget `ℓ∞` et ce
  qui survit à la purification (JPEG, passe-bas DCT, débruitage), plus l'attaque WB-LF contrainte au
  sous-espace conservé. Répond à la question ouverte de Hong et al. (2024).
