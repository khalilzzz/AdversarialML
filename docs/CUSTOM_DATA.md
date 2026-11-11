# Empoisonner tes propres données

Ce document explique comment lancer l'attaque **hors de l'article** : ta base de données, ta cible,
tes paramètres. Rien n'est codé en dur pour CIFAR-10 ou ImageNet — un mode `dataset: custom` lit un
ImageFolder quelconque, à n'importe quelle résolution et n'importe quel nombre de classes.

Si tu n'as pas encore lu `docs/GUIDE.md`, lis-le d'abord : il explique les deux réseaux (surrogate
gelé contre victime from-scratch) et les deux métriques (succès sur une image contre précision
globale). Tout ce qui suit suppose ce modèle mental.

---

## 1. Disposer tes données

Un arbre `torchvision.datasets.ImageFolder`, un dossier par classe, séparé train/val :

```
<data_path>/
  train/
    classe_a/ img001.jpg img002.jpg ...
    classe_b/ ...
  val/
    classe_a/ ...        # doit contenir ton image cible
    classe_b/ ...
```

- Les noms de classes sont libres ; l'index d'une classe est sa **position alphabétique**
  (`classe_a` → 0, `classe_b` → 1, …). C'est ce qui donne `adv_class`.
- Formats d'image : tout ce que PIL lit (jpg, png, …). Tailles hétérogènes acceptées : chaque image
  est ramenée à `image_size × image_size` par `Resize` + `CenterCrop`.
- Le dossier `val/` doit contenir l'image que tu veux faire mal classer.

---

## 2. Écrire la config

Copie `configs/custom/my_attack.yaml` et ajuste. Les champs qui comptent :

| Champ | Rôle |
|---|---|
| `dataset: custom` | active le mode ImageFolder générique |
| `data_path` | le dossier qui contient `train/` et `val/` |
| `image_size` | côté carré imposé à toutes les images (ex. 64) |
| `data_mean` / `data_std` | omis → calculés une fois sur `train/` et mis en cache dans `<data_path>/custom_stats.json` |
| `augment_shift` | budget de translation en pixels ; omis → `round(image_size/8)` |
| `num_classes_override` | omis → compté depuis `train/` |
| `target_index` / `adv_class` | ta cible explicite (voir §3) |
| `budget`, `eps`, `targets` | le modèle de menace : fraction empoisonnée, borne `l_inf` (0–255), nombre de cibles |
| `net`, `width`, `epochs`, `victim_runs` | architecture et recette d'entraînement |

Routage d'architecture : en mode custom, `image_size ≤ 64` utilise le stem CIFAR des ResNets (3×3,
pas de max-pool) ; au-dessus, les architectures torchvision standard. Tu n'as rien à faire, c'est
automatique.

---

## 3. Choisir la cible

L'ordre ImageFolder est déterministe (classes triées par nom, puis fichiers triés par nom). Pour
trouver l'index de ton image cible dans le jeu de validation :

```bash
python scripts/poison_custom.py --config configs/custom/my_attack.yaml --list-targets
```

Sortie : `index | classe | chemin`. Repère la ligne de ton image, note son `index`. C'est ton
`--target-index`. Choisis ensuite la classe cible adverse `--adv-class` (la classe dans laquelle tu
veux que l'image soit classée, en index alphabétique).

---

## 4. Lancer l'attaque

**Cible explicite, un cas, contrôle total :**

```bash
python scripts/poison_custom.py --config configs/custom/my_attack.yaml \
       --target-index 1234 --adv-class 3 --export png
```

Ce qui se passe : entraîne/charge le surrogate propre, brasse les poisons (Algorithme 1), exporte les
images empoisonnées, puis entraîne `victim_runs` victimes from-scratch sur les données empoisonnées
et mesure le succès. Les poisons sont tirés de la classe `adv_class`, **étiquettes intactes**.

Sorties :
- `poisons/<name>/` — images empoisonnées (PNG : `clean/` et `poisoned/`).
- `results/<name>/custom_result.json` — `brew_loss`, succès, précision propre, invariant `l_inf`.

**Brasser sans entraîner de victime** (juste regarder/exporter les poisons) : ajoute `--brew-only`.

**Protocole aléatoire multi-cas sur tes données** (façon papier, sans cible fixée) : laisse
`target_index` et `adv_class` à `null` dans la config, et lance plusieurs cas :

```bash
python scripts/run_experiment.py --config configs/custom/my_attack.yaml \
       --set cases=10 --set victim_runs=8
python scripts/aggregate.py
```

Le code tire alors les paires de classes et les cibles depuis une graine (comme
`random_construction` de l'auteur), de façon déterministe.

---

## 5. Lire la sortie

Trois signaux, par ordre de fiabilité à petite échelle :

1. **`l_inf after quantization ≤ eps`** — l'invariant de menace. Après passage en 8 bits, aucune
   perturbation ne dépasse `eps/255`. Doit toujours tenir.
2. **`B < 1`** (`brew_loss`) — le cosinus entre le gradient des poisons et celui de la cible est
   positif, donc les gradients sont alignés : l'attaque a convergé. C'est le signe que le brewing a
   marché, indépendamment de la victime.
3. **`poison success`** — avec **une** cible et peu de victimes, il est binaire (0 % ou 100 %).
   C'est la nature d'une attaque mono-cible, pas un défaut. Pour un chiffre stable, utilise le mode
   multi-cas, ou augmente `victim_runs`.

La **précision propre** doit rester proche de celle d'un modèle non empoisonné : l'attaque est une
attaque d'intégrité, elle ne dégrade pas la performance globale.

---

## 6. Tous tes paramètres

Chaque valeur est surchargeable en ligne de commande via `--set clé=valeur`, enregistrée dans le JSON
de résultat, et marque le run comme non conforme au protocole publié. Exemples :

```bash
# plus fort : budget 4%, eps 32, 16 pas de plus
--set budget=0.04 --set eps=32 --set attack_steps=250

# objectif euclidien au lieu du cosinus
--set objective=euclidean

# brewing sur une architecture, victime sur une autre (transfert)
--set net=ResNet18 --set victim_net=MobileNetV2

# ensemble de 4 surrogates
--set ensemble=4
```

Les défauts de brewing (`tau=0.1`, `restarts=8`, `attack_steps=250`) sont ceux de l'article et
restent de bons points de départ sur tes données.
