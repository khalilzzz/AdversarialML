# Pistes de recherche autour de Witches' Brew — présentation

Nov 11, 2026 · @Khalil

## Résumé exécutif

Recommandation : présenter la **Piste 1** comme axe principal (attaque + théorie + expériences), la **Piste 2** comme axe secondaire bon marché, la Piste 3 en une phrase. Les deux premières répondent à des questions ouvertes posées explicitement par les auteurs des articles lus, et aucune n'a de publication équivalente trouvée sur le web (vérification du 11 nov. 2026).

| Piste | Idée en une phrase | Type de contribution | Coût calcul | Nouveauté vérifiée |
| --- | --- | --- | --- | --- |
| 1. WB conscient de la purification | Rendre le gradient matching robuste aux défenses par débruitage (JPEG, EBM, diffusion) en contraignant la perturbation au sous-espace que la purification conserve (basses fréquences / variété des images naturelles). | Attaque adaptative + résultat théorique sous contrainte linéaire + expériences | Moyen (CIFAR-10, ResNet-18, code WB existant) | Appelée explicitement par Hong–Carlini–Kurakin (ICML 2024) ; aucune attaque ciblée adaptative trouvée |
| 2. WB contre DPA (agrégation) | Exploiter le hash public de DPA pour concentrer les poisons dans peu de partitions, et tester si le gradient matching souffre du « poisoning overfitting ». | Attaque + borne d'allocation + test d'une hypothèse empirique | Faible (modèles de base entraînés sur 1 000–5 000 images) | Listée comme direction future par Wang & Feizi (2023) ; rien trouvé |
| 3. Trajectory matching | Remplacer l'alignement d'un seul gradient (WB) par l'alignement de la trajectoire d'entraînement (idée de FLIP / distillation de données). | Extension algorithmique | Élevé | Fait pour les labels (FLIP, NeurIPS 2023), pas pour les poisons clean-label |

Ce qui est écarté : rendre WB robuste aux variations de la cible (flip, recadrage) — déjà publié en déc. 2024 (Chen et al., arXiv 2412.03908).

## Ce que disent les 13 articles, vus depuis Witches' Brew

Le fil conducteur : depuis 2021, les défenses par **purification** (JPEG, EBM, diffusion) et par **agrégation** (DPA) réduisent le succès de Witches' Brew (WB) à 0–10 %, et leurs auteurs demandent eux-mêmes des attaques adaptatives plus fortes. C'est là que se trouvent les questions ouvertes.

| Article | Nature | Ce qu'il apporte sur WB | Question ouverte qu'il laisse |
| --- | --- | --- | --- |
| [Hong, Carlini, Kurakin — Certified Robustness via Diffusion Denoising](https://arxiv.org/abs/2403.11981) (ICML 2024) | Défense certifiée (ℓ₂) | Un débruitage par DDPM sur tout le train set fait passer WB de 65–71 % à 3–10 % de succès (σ = 0,25, from scratch). Le certificat ne couvre qu'un rayon ℓ₂ ≈ 0,25–0,5, très loin du budget réel de WB. | « Current attacks are weak … need to study the worst-case poisoning adversary who remains effective under our defense » ; attaques adaptatives contre FrieNDs laissées en travail futur. |
| [PureGen](https://arxiv.org/abs/2405.18627) / [PureEBM](https://arxiv.org/abs/2405.19376) (2024) | Défense par purification générative | Langevin sur EBM (150 pas) ou diffusion partielle (75 pas) : WB 1 % passe de 44 % à 0–1 %. Point clé : **JPEG seul fait aussi 0 %** (Table 1). Les poisons ont une énergie EBM nettement plus haute que les images propres. | Aucune attaque adaptative testée ; l'énergie EBM comme détecteur suggère une contrainte « basse énergie » côté attaquant. |
| [SAPA — Sharpness-Aware Data Poisoning](https://arxiv.org/abs/2305.14851) (ICLR 2024) | Amélioration d'attaque | Remplace le gradient cible ∇Q(θ\*) par ∇Q(θ\* + v̂) au pire voisin (style SAM, ρ = 0,05) : WB passe de 72,9 → 80,1 % (ε = 16) et surtout 10,3 → 16,7 % à ε = 4. Moins cher que l'ensemble + réinitialisation. Code public. | Brique réutilisable : compatible avec n'importe quelle contrainte ajoutée sur δ. |
| [Wang & Feizi — Practical Aspects of Aggregation Defenses](https://arxiv.org/abs/2306.16415) (2023) | Analyse de DPA | DPA est empiriquement robuste bien au-delà de son certificat grâce au **poisoning overfitting** : à pourcentage de poison fixe, réduire la taille du train set réduit le succès. Hypothèse implicite : les poisons sont répartis uniformément entre partitions. | « Designing advanced attacks that avoid poisoning overfitting to prevent aggregation defenses from enhancing robustness ». |
| [Temporal Robustness](https://arxiv.org/abs/2302.03684) (NeurIPS 2023) | Nouveau modèle de menace | Remplace le budget en nombre d'échantillons par earliness/duration ; agrégation temporelle certifiée. | Peu lié à WB directement ; montre que le « budget » est une notion à repenser. |
| [Schwarzschild et al. — Just How Toxic](https://arxiv.org/abs/2006.12557) (ICML 2021) | Benchmark | WB est la seule attaque qui marche from scratch ; mais toutes les attaques sont fragiles : SGD, augmentation, taille du dataset, et **un simple flip de la cible** les cassent. | Le succès n'est pas invariant à la taille du dataset à % fixe (non testé pour WB). |
| [Pick your Poison](https://arxiv.org/abs/2305.09671) (2023) | Cadre d'évaluation | Formalise le compromis **robustesse vs détectabilité** (sur-empoisonner = robuste mais détectable). | Jamais tracé pour WB : courbe (budget, ε) → (succès, détectabilité par énergie EBM). |
| [FLIP — Label Poisoning is All You Need](https://arxiv.org/abs/2310.18933) (NeurIPS 2023) | Attaque label-only | Trajectory matching (distillation de données) pour choisir des labels à flipper : backdoor à 99 % avec 2 % de labels. | Le trajectory matching n'a pas été appliqué aux poisons clean-label ciblés. |
| [Koh, Steinhardt, Liang — Stronger attacks break sanitization](https://arxiv.org/abs/1811.00741) (ML 2022) | Attaques adaptatives (modèles linéaires) | Deux principes : **concentrer** les poisons, et formuler l'attaque comme optimisation **sous contrainte d'évasion** du détecteur. | Ces deux principes sont exactement ce que les Pistes 1 et 2 transposent aux DNN. |
| [Geiping et al. — What doesn't kill you](https://arxiv.org/abs/2102.13624) (2021) | Défense par adversarial training | Entraîner avec des poisons générés à la volée ; meilleur compromis que DP-SGD. | Coûteux ; les auteurs de la défense certifiée le battent en utilité. |
| [Wild Patterns Reloaded](https://arxiv.org/abs/2205.01992) (ACM CSUR 2023) | Survey | Deux limites majeures : modèles de menace irréalistes (cible exacte connue, gros budgets) et coût du bilevel. | Appelle à des attaques « adaptive to the presence of a defender ». |
| [Data Poisoning in Deep Learning: A Survey](https://arxiv.org/abs/2503.22759) (2025) | Survey | Taxonomie ; directions : attaques plus furtives et adaptatives, transférabilité. | Générique. |

Note : les deux fichiers « Sharpness-aware » uploadés sont identiques (même md5), d'où 13 articles et non 14.

## Piste 1 — Witches' Brew conscient de la purification (recommandée)

Thèse : les défenses par débruitage battent WB non pas grâce à leur certificat, mais parce que WB dépense son budget ℓ∞ dans le sous-espace (hautes fréquences, hors variété) que la purification efface. Une attaque qui contraint la perturbation au sous-espace conservé devrait survivre ; on peut le démontrer dans le cas linéaire et le tester sur CIFAR-10.

### Pourquoi c'est une vraie question ouverte

1. Hong, Carlini et Kurakin (ICML 2024) concluent littéralement : les attaques actuelles sont faibles, elles n'exploitent pas le budget ε = 16/255, et il faut étudier l'adversaire du pire cas qui reste efficace sous leur défense.
2. Leur certificat est **vide** au budget de WB. Pour une image CIFAR-10 (d = 3 072) et ε = 16/255, chaque poison a ‖δ‖₂ ≤ 3,48 ; pour r = 500 poisons, le critère du Corollaire 3.4 demande √(Σ‖δᵢ‖₂²) < σ(Φ⁻¹(p\_A) − Φ⁻¹(p\_B))/2, soit au mieux ≈ 0,6 pour σ = 0,25, alors que l'attaque réelle vaut √500 × 3,48 ≈ 78. La défense marche donc empiriquement \~100× au-delà de ce qu'elle garantit : il y a un mécanisme non expliqué à comprendre, et donc à exploiter.
3. PureGen montre que **JPEG seul** ramène WB à 0 % (Table 1). JPEG est un filtre passe-bas quantifié : si WB meurt sous JPEG, c'est que ses perturbations vivent dans les hautes fréquences DCT.
4. Vérification web (11 nov. 2026) : les attaques adaptatives contre la purification par diffusion existent pour l'évasion (DiffAttack, DiffBreak) et pour les poisons indiscriminés (ECLIPSE), pas pour le poisoning ciblé clean-label. Les perturbations basse fréquence sont connues pour transférer mieux en évasion (Sharma et al., 2019) — jamais appliquées au gradient matching.

### Formulation

WB résout, à θ\* fixé (modèle propre pré-entraîné) :

```latex
\min_{\|\delta_i\|_\infty \le \varepsilon}\; 1 - \frac{\big\langle \nabla_\theta \ell(\theta^*; x_t, y_{adv}),\; \sum_i \nabla_\theta \ell(\theta^*; x_i + \delta_i, y_i) \big\rangle}{\|\nabla_\theta \ell(\theta^*; x_t, y_{adv})\|\;\|\sum_i \nabla_\theta \ell(\theta^*; x_i + \delta_i, y_i)\|}
```

Version consciente de la purification Ψ (JPEG, débruiteur DDPM à un pas, K pas de Langevin EBM), avec espérance sur l'aléa ξ de Ψ (EOT) :

```latex
\min_{\|\delta_i\|_\infty \le \varepsilon}\; 1 - \cos\!\Big( \nabla_\theta \ell(\theta^*; x_t, y_{adv}),\; \mathbb{E}_\xi \sum_i \nabla_\theta \ell\big(\theta^*; \Psi_\xi(x_i + \delta_i), y_i\big) \Big)
```

Deux variantes moins chères et agnostiques à la défense :

- **WB-LF (spectrale)** : paramétrer δᵢ = IDCT(M ⊙ cᵢ) avec M un masque basse fréquence (n × n premiers coefficients, n ∈ {8, 12, 16} sur 32), puis projeter sur la boule ℓ∞. Pas de backprop à travers Ψ.
- **WB-énergie** : ajouter la pénalité λ·G(xᵢ + δᵢ) avec G l'énergie d'un EBM public, pour rester dans la variété des images naturelles (PureGen détecte les poisons justement par leur énergie élevée).

### Résultat théorique à présenter (contrainte linéaire)

Proposition (cas linéaire / NTK). Soit f(x) = wᵀx avec perte logistique ; alors ∇\_w ℓ(x, y) = r(x, y)·x avec r scalaire, donc la direction du gradient d'un poison est celle de xᵢ + δᵢ et celle de la cible est x\_t. Si le défenseur applique un projecteur orthogonal P sur un sous-espace S (passe-bas, PCA, etc.), l'alignement atteignable vérifie, pour tout δᵢ :

```latex
\cos\big(P(x_i+\delta_i),\, x_t\big) \;=\; \frac{\langle P(x_i+\delta_i),\, P x_t\rangle}{\|P(x_i+\delta_i)\|\,\|x_t\|} \;\le\; \frac{\|P x_t\|}{\|x_t\|} \;=:\; \rho_S(x_t)
```

avec égalité si et seulement si P(xᵢ + δᵢ) est colinéaire à P x\_t. Conséquences : (i) la composante de δᵢ dans S⊥ est entièrement perdue ; (ii) l'alignement maximal est borné par la fraction d'énergie de la cible dans S ; (iii) à budget ℓ∞ fixé, l'attaquant optimal cherche δᵢ ∈ S ∩ {‖δ‖∞ ≤ ε} (projections alternées / Dykstra). Pour un réseau non linéaire, la règle de la chaîne donne la version au premier ordre : ∇\_θ ℓ(θ; Ψ(x+δ)) ne dépend de δ qu'à travers J\_Ψ(x)·δ, donc toute composante de δ dans le noyau de la jacobienne de la purification est gaspillée. C'est un résultat court, démontrable au tableau, qui relie directement le cours de traitement du signal (Fourier/DCT) au cours d'adversarial ML.

Extension possible : avec le débruitage gaussien à un pas (x → D(x + ση)), le débruiteur optimal est linéaire par morceaux près de la variété (Tweedie) et P devient la projection sur le plan tangent local de la variété des images ; ρ\_S(x\_t) devient « la fraction du gradient cible expliquée par la variété », mesurable empiriquement.

### Plan d'expériences (CIFAR-10, ResNet-18, code WB officiel)

| Étape | Quoi | Durée | Résultat attendu / critère |
| --- | --- | --- | --- |
| E0 — diagnostic | Prendre des poisons WB existants (ε = 16, 1 %). Mesurer le spectre DCT de δ, la fraction d'énergie de δ qui survit à JPEG-75, au DDPM σ = 0,25 et à 150 pas d'EBM, et l'alignement de gradient conservé après Ψ. | 1–2 jours | Figure : énergie de δ concentrée en hautes fréquences, alignement effondré après Ψ. C'est déjà un résultat présentable. |
| E1 — WB-LF | Implémenter la paramétrisation DCT masquée dans le code WB. Évaluer sans défense puis sous JPEG (le baseline à 0 % de PureGen). | 1 semaine | Succès > 20 % sous JPEG avec n = 12 ou 16 serait une contribution nette. |
| E2 — WB-Ψ | BPDA/EOT à travers un JPEG différentiable puis à travers le DDPM CIFAR-10 de Ho et al. à un pas (setting exact de Hong et al.). Comparer à leur Table 2 : WB 65 % → 3–10 % à σ = 0,25. | 2 semaines | Remonter au-dessus de 30 % à σ = 0,25 casserait le baseline qu'ils recommandent. |
| E3 — compromis | Tracer succès vs détectabilité (AUC de l'énergie EBM) et vs visibilité (LPIPS/SSIM) en fonction de ε et de la coupure fréquentielle, à la manière de Pick your Poison. | 1 semaine | La courbe elle-même est la contribution ; combiner avec SAPA si le temps le permet. |

### Pièges à anticiper

- À ℓ∞ égal, une perturbation basse fréquence est plus visible (Sharma et al.) : toujours reporter LPIPS/SSIM et des exemples visuels.
- Gradients obfusqués (Athalye et al., 2018) : utiliser EOT sur l'aléa de Ψ et vérifier que l'attaque reste efficace quand le défenseur change de graine.
- Coût mémoire de la backprop à travers un débruiteur : rester à un pas de diffusion (c'est le setting certifié) ; DiffAttack/DiffBreak donnent les astuces si besoin.

## Piste 2 — Witches' Brew contre les défenses par agrégation (DPA)

Thèse : la robustesse **empirique** de DPA (bien au-delà de son certificat) repose sur deux hypothèses implicites — répartition uniforme des poisons entre partitions, et « poisoning overfitting » — qu'un attaquant par gradient matching peut violer toutes les deux. Les expériences sont très bon marché parce que les modèles de base de DPA s'entraînent sur de petites partitions.

### Rappel de DPA et de l'observation de Wang & Feizi

DPA partitionne le train set en k sous-ensembles par un hash déterministe h(x, y) (dans l'implémentation d'origine : somme des pixels mod k), entraîne un modèle par partition et vote. Le certificat garantit la prédiction tant que le nombre de poisons est inférieur à la moitié de la marge de votes, soit O(k) — une dizaine d'échantillons sur CIFAR-10. Pourtant, Wang & Feizi (2023, Table 6) montrent que DPA réduit le succès d'attaques avec 32 à 3 843 poisons, et l'expliquent par le poisoning overfitting : à pourcentage fixé, moins d'échantillons absolus ⇒ l'attaque généralise moins. Ils écrivent que si cette hypothèse tombe, DPA n'apporte aucune robustesse empirique supplémentaire, et proposent comme travail futur de concevoir des attaques qui l'évitent.

### Les deux leviers de l'attaquant

1. **Placement conscient du hash.** Le hash est public et ne dépend que de l'échantillon lui-même. L'attaquant contrôle donc la partition de chaque poison : pour la somme des pixels mod k, il suffit d'ajuster un pixel de moins de k/255 — négligeable devant ε = 16/255. Au lieu de 500 poisons répartis sur k partitions (1 % chacune), il peut mettre 500/m poisons dans m partitions choisies : avec k = 50 et m = 10, chaque partition attaquée contient 5 % de poisons sur 1 000 images. Le certificat n'est pas violé (il couvre déjà le pire placement) ; c'est la robustesse empirique qui s'effondre si l'attaque par partition réussit.
2. **Poisons spécifiques par partition.** Chaque modèle de base est entraîné sur \~1 000 images : l'attaquant peut simuler exactement ce modèle de substitution et faire du gradient matching dessus, ce qui est bien plus fidèle que le modèle unique de WB classique.

### Résultat théorique à présenter (allocation optimale)

Soit p(ρ) la probabilité qu'un modèle de base soit retourné quand sa partition contient une fraction ρ de poisons (courbe mesurable), B le budget total, n/k la taille d'une partition, et M la marge de votes de la cible (nombre de modèles à retourner = ⌈(M+1)/2⌉ =: m₀). L'attaquant choisit m partitions et y met B/m poisons chacune, soit ρ = Bk/(mn). Le succès est P\[Bin(m, p(Bk/(mn))) ≥ m₀\]. Deux résultats simples :

```latex
\text{(i)}\;\; \text{si } p \text{ est concave, l'allocation uniforme sur } m \ge m_0 \text{ partitions est optimale à } m \text{ fixé ; }\qquad \text{(ii)}\;\; m^\star = \arg\max_{m \ge m_0} \; \mathbb{P}\big[\mathrm{Bin}(m, p(Bk/(mn))) \ge m_0\big]
```

Comparer m\* à la répartition uniforme (m = k) donne exactement le facteur de robustesse empirique que DPA perd face à un attaquant informé. Si p(ρ) est une sigmoïde en ρ (ce que SAPA et le benchmark de Schwarzschild suggèrent : le succès de WB saute entre 0,2 % et 1 %), la concentration gagne énormément. On peut aussi relier m₀ à la marge certifiée : le certificat de DPA est serré précisément pour l'attaquant qui contrôle le hash.

### Plan d'expériences (tout tient sur un GPU modeste)

| Étape | Quoi | Durée | Critère |
| --- | --- | --- | --- |
| E0 — poisoning overfitting pour WB | WB à 1 % sur des sous-ensembles de CIFAR-10 de taille 50k, 25k, 10k, 5k, 2k (modèle de base de taille adaptée). Schwarzschild et al. l'ont fait pour FC/CP/CLBD/HTBD (Fig. 2), jamais pour WB. | 3–4 jours | Si le succès de WB **ne baisse pas** quand la taille diminue, l'hypothèse de Wang & Feizi est fausse pour le gradient matching : résultat publiable en soi. |
| E1 — mesurer p(ρ) | Sur des partitions de 1 000 images, succès de WB pour ρ ∈ {1, 2, 5, 10 %}. | 2 jours | Courbe p(ρ) ; alimente la borne. |
| E2 — attaque hash-aware | DPA k ∈ {10, 50} sur CIFAR-10 ; comparer placement aléatoire vs concentré sur m\* partitions (poisons crafted sur le modèle de substitution de chaque partition). | 1 semaine | Succès de l'agrégat bien au-dessus du placement aléatoire à budget égal. |
| E3 — contre-mesure | Hash secret (keyed) ou aléatoire : montre que la robustesse empirique de DPA dépend du secret du hash, ce que la littérature ne dit pas. | 2 jours | Discussion défense. |

### Pourquoi c'est un bon complément à la Piste 1

Les deux pistes transposent aux DNN les deux principes de Koh, Steinhardt & Liang (2022) : concentrer les poisons (Piste 2) et optimiser sous contrainte d'évasion du défenseur (Piste 1). Présentées ensemble, elles racontent une seule histoire : « WB est évalué contre des défenses qui supposent un attaquant non adaptatif ».

## Piste 3 (bonus) — du gradient matching au trajectory matching

WB aligne un seul gradient, calculé à un modèle propre déjà entraîné θ\*, alors que la victime part d'une initialisation aléatoire : l'alignement doit tenir tout au long de l'entraînement, et il s'effondre dès que ε baisse (10 % de succès à ε = 4/255). FLIP (NeurIPS 2023) résout le même problème pour des labels en alignant la **trajectoire** d'un expert empoisonné : à un checkpoint θ\_k tiré au hasard, on compare le pas fait sur le batch empoisonné à celui fait sur le batch à poisons clean-label, avec la perte normalisée ‖θ\_{k+1} − φ\_{k+1}‖² / ‖θ\_{k+1} − θ\_k‖².

Transposition : l'expert est un modèle entraîné avec la cible mal étiquetée insérée (l'attaque « non contrainte » que WB cherche à imiter, cf. leur section 3.1) ; on optimise les δᵢ pour que les pas sur (xᵢ + δᵢ, yᵢ) reproduisent ceux de l'expert à plusieurs checkpoints. Lien théorique à mettre en avant : le cosinus de WB est l'approximation au premier ordre, à un seul point θ\*, de cet objectif ; le trajectory matching contrôle l'erreur accumulée sur K pas, ce qui est exactement la « retraining uncertainty » que SAPA attaque par un autre moyen (pire voisin de θ\*).

- Avantage : mécanisme différent de SAPA et de l'ensemble + réinitialisation de MetaPoison, donc combinable.
- Coût : stocker des checkpoints (K ≤ 20 époques suffisent chez FLIP) et faire une backprop à travers un pas de SGD par itération ; plus lourd que WB mais bien moins que MetaPoison.
- État de l'art vérifié : trajectory matching utilisé pour les labels (FLIP) et la distillation de données ; pas trouvé pour les poisons clean-label ciblés. Zheng & Li (INFOCOM 2021) alignent une mise à jour de premier ordre, pas une trajectoire.

À garder pour une seconde phase : c'est la piste la plus lourde en calcul et la moins reliée aux articles fournis.

## Pistes écartées parce que déjà faites

Les mentionner devant le professeur montre que la veille a été faite ; chacune pointe vers l'article qui l'a déjà traitée.

| Idée tentante | Déjà fait par | Ce qu'il reste éventuellement |
| --- | --- | --- |
| WB robuste aux variations de la cible (flip, recadrage, objet physique vu sous plusieurs angles) — la faiblesse pointée par Schwarzschild et al. et par Wild Patterns Reloaded | [Chen et al., « Can Targeted Clean-Label Poisoning Attacks Generalize? »](https://arxiv.org/abs/2412.03908) (déc. 2024) : montre que le cosinus de WB généralise mal et ajoute un terme de distance euclidienne sur les gradients. | Rien de central ; au mieux combiner avec la Piste 1. |
| Rendre WB robuste à l'incertitude du ré-entraînement (init, architecture, optimiseur) | SAPA (ICLR 2024) via la sharpness ; MetaPoison via ensemble + réinitialisation. | Combiner SAPA avec une contrainte de purification (Piste 1, E3). |
| Attaque adaptative contre la purification par diffusion | Faite pour l'évasion (DiffAttack ICLR 2024, DiffBreak 2025) et pour les poisons indiscriminés (ECLIPSE, ECCV 2024). | **Pas faite pour le poisoning ciblé clean-label** : c'est précisément la Piste 1. |
| Poisons basse fréquence | Faits pour l'évasion (Sharma et al., 2019) et pour des triggers de backdoor (Checkerboard, 2026). | Pas pour le gradient matching : Piste 1, variante WB-LF. |
| Attaque label-only ciblée | FLIP note que le poisoning ciblé par flips de labels est considéré comme facile (plus proches voisins de la cible). | Peu de valeur ajoutée. |
| Tester WB contre DPA de façon naïve | Baracaldo et al. (cité par Wang & Feizi) ont évalué des attaques clean-label contre DPA sans placement adaptatif. | Le placement conscient du hash et le test du poisoning overfitting pour WB restent ouverts : Piste 2. |

## Ce que dire demain, et le plan des semaines suivantes

### Script de 5 minutes

1. **Constat (1 min).** « J'ai lu les 13 articles voisins de Witches' Brew sur Connected Papers. Depuis 2023, trois familles de défenses ramènent WB à 0–10 % de succès : purification (JPEG, EBM, diffusion), agrégation (DPA), adversarial training. Les auteurs de la défense certifiée par diffusion (Carlini et al.) écrivent eux-mêmes que les attaques actuelles sont faibles et qu'il faut l'adversaire du pire cas. »
2. **L'anomalie (1 min).** « Leur certificat ne couvre qu'un rayon ℓ₂ d'environ 0,5, alors que WB à ε = 16/255 avec 500 poisons représente une perturbation ℓ₂ totale d'environ 78. La défense marche 100 fois au-delà de ce qu'elle prouve. Mon hypothèse : WB dépense son budget dans les hautes fréquences, que le débruitage efface. Indice : JPEG seul suffit à tuer WB (PureGen, Table 1). »
3. **La piste principale (2 min).** Montrer la proposition linéaire (alignement ≤ ‖Px\_t‖/‖x\_t‖, composante hors sous-espace gaspillée) et la formulation WB-Ψ / WB-LF. Annoncer E0 (diagnostic spectral, 2 jours) comme premier livrable, puis E1 sous JPEG, puis E2 contre le DDPM à un pas (le setting exact de l'article certifié).
4. **La piste secondaire (1 min).** « DPA n'est robuste empiriquement que si les poisons sont répartis uniformément et si l'attaque souffre de poisoning overfitting. Le hash de DPA est public : je peux choisir la partition de chaque poison. Wang & Feizi listent cette attaque comme travail futur. Les expériences sont très bon marché (modèles sur 1 000 images). »
5. **Question au professeur.** Préfère-t-il un résultat théorique propre sur la Piste 1 (cas linéaire + débruiteur gaussien) ou une évaluation empirique complète ? Cela fixe le ratio théorie/expériences du rapport.

### Si on vous demande « qu'avez-vous déjà fait ? »

Réponse honnête et défendable : lecture et cartographie des 13 articles, identification de deux questions ouvertes posées explicitement par les auteurs, vérification de nouveauté sur le web (une piste écartée car publiée en déc. 2024), un résultat théorique simple déjà formulé, et un plan d'expériences dont la première étape tient en deux jours sur le code public de WB.

### Plan des prochaines semaines

- [ ] Semaine 1 : installer le code officiel de WB ; reproduire un succès de base (ε = 16, 1 %, ResNet-18, \~10 cibles) ; E0 de la Piste 1 (spectre DCT des poisons, énergie survivant à JPEG / DDPM / EBM).
- [ ] Semaine 2 : WB-LF (paramétrisation DCT masquée) sous JPEG ; en parallèle E0 de la Piste 2 (WB à % fixe vs taille du dataset).
- [ ] Semaine 3 : WB-Ψ à travers le DDPM à un pas (setting de Hong et al.) ; rédiger la proposition et son extension au débruiteur gaussien.
- [ ] Semaine 4 : attaque hash-aware contre DPA (k = 10 puis 50) ; courbe succès vs détectabilité par énergie ; rédaction.

Règle de présentation des résultats, tirée du benchmark de Schwarzschild et al. : au moins 20–50 cibles tirées au hasard (classes de base et cible aléatoires), SGD + augmentation standard, intervalle de confiance reporté.

## Références et code à réutiliser

### Code

| Dépôt | Usage |
| --- | --- |
| [JonasGeiping/poisoning-gradient-matching](https://github.com/JonasGeiping/poisoning-gradient-matching) | Code officiel de Witches' Brew ; base de toutes les expériences. |
| [PengfeiHePower/SAPA](https://github.com/PengfeiHePower/SAPA) | Variante sharpness-aware du gradient matching (brique pour E3). |
| [SunayBhat1/PureGen\_PoisonDefense](https://github.com/SunayBhat1/PureGen_PoisonDefense) | EBM et DDPM de purification entraînés, pipeline d'évaluation, poisons WB déjà générés. |
| [aks2203/poisoning-benchmark](https://github.com/aks2203/poisoning-benchmark) | Benchmark de Schwarzschild et al. (protocole d'évaluation, modèles pré-entraînés). |
| DPA : implémentation de Levine & Feizi (ICLR 2021) | Partitionnement par hash, vote, certificat. |
| DiffJPEG (PyTorch) | JPEG différentiable pour WB-Ψ. |
| DDPM CIFAR-10 de Ho et al. (2020) | Débruiteur à un pas, setting de Hong et al. |

### Articles lus (fournis)

- Hong, Carlini, Kurakin. Certified Robustness to Clean-Label Poisoning Using Diffusion Denoising. ICML 2024. [arXiv:2403.11981](https://arxiv.org/abs/2403.11981)
- Bhat, Jiang, Pooladzandi et al. PureGen. 2024. [arXiv:2405.18627](https://arxiv.org/abs/2405.18627) ; PureEBM. [arXiv:2405.19376](https://arxiv.org/abs/2405.19376)
- He et al. Sharpness-Aware Data Poisoning Attack. ICLR 2024. [arXiv:2305.14851](https://arxiv.org/abs/2305.14851)
- Wang, Feizi. On Practical Aspects of Aggregation Defenses. 2023. [arXiv:2306.16415](https://arxiv.org/abs/2306.16415)
- Wang, Feizi. Temporal Robustness against Data Poisoning. NeurIPS 2023. [arXiv:2302.03684](https://arxiv.org/abs/2302.03684)
- Schwarzschild et al. Just How Toxic is Data Poisoning? ICML 2021. [arXiv:2006.12557](https://arxiv.org/abs/2006.12557)
- Lukas, Kerschbaum. Pick your Poison. 2023. [arXiv:2305.09671](https://arxiv.org/abs/2305.09671)
- Jha, Hayase, Oh. Label Poisoning is All You Need. NeurIPS 2023. [arXiv:2310.18933](https://arxiv.org/abs/2310.18933)
- Koh, Steinhardt, Liang. Stronger Data Poisoning Attacks Break Data Sanitization Defenses. Machine Learning 2022. [arXiv:1811.00741](https://arxiv.org/abs/1811.00741)
- Geiping et al. What Doesn't Kill You Makes You Robust(er). 2021. [arXiv:2102.13624](https://arxiv.org/abs/2102.13624)
- Cinà et al. Wild Patterns Reloaded. ACM CSUR 2023. [arXiv:2205.01992](https://arxiv.org/abs/2205.01992)
- Zhao et al. Data Poisoning in Deep Learning: A Survey. 2025. [arXiv:2503.22759](https://arxiv.org/abs/2503.22759)

### Trouvés pendant la vérification de nouveauté

- Chen et al. Can Targeted Clean-Label Poisoning Attacks Generalize? déc. 2024. [arXiv:2412.03908](https://arxiv.org/abs/2412.03908) — ferme la piste « cible robuste ».
- Wang et al. ECLIPSE: Expunging Clean-label Indiscriminate Poisons via Sparse Diffusion Purification. ECCV 2024. [arXiv:2406.15093](https://arxiv.org/abs/2406.15093) — attaque adaptative, mais indiscriminée.
- Kang et al. DiffAttack. NeurIPS 2023 ; Kassis et al. DiffBreak. 2025. [arXiv:2411.16598](https://arxiv.org/abs/2411.16598) — attaques adaptatives contre la purification, évasion seulement.
- Sharma, Ding, Brubaker. On the Effectiveness of Low Frequency Perturbations. IJCAI 2019. [arXiv:1903.00073](https://arxiv.org/abs/1903.00073) — masques DCT réutilisables pour WB-LF.
- Levine, Feizi. Deep Partition Aggregation. ICLR 2021. [arXiv:2006.14768](https://arxiv.org/abs/2006.14768)
- Geiping et al. Witches' Brew. ICLR 2021. [arXiv:2009.02276](https://arxiv.org/abs/2009.02276)
