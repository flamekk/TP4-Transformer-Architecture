# Méthodologie — Architecture Transformer

## 1. Présentation générale

Ce projet étudie les architectures Transformer et leurs applications à différentes tâches de traitement de séquences.

Le travail est organisé autour de trois axes pratiques :

- Prévision de séries temporelles avec un Transformer.
- Génération autorégressive de texte avec un Mini-GPT.
- Prédiction de mots masqués avec un Mini-BERT.

Une première partie du notebook compare également les architectures RNN, LSTM et Transformer sur une série temporelle synthétique.

---

## 2. Comparaison des architectures séquentielles

### 2.1 Données synthétiques

Une série temporelle sinusoïdale est utilisée pour étudier la capacité des architectures à apprendre des dépendances temporelles.

### 2.2 Architectures étudiées

Trois modèles sont comparés :

- **RNN** : traite les éléments d'une séquence de manière récurrente.
- **LSTM** : utilise des mécanismes de portes pour mieux gérer les dépendances à long terme.
- **Transformer** : exploite le mécanisme d'attention pour modéliser les relations entre les éléments d'une séquence.

Les prédictions sont comparées aux valeurs réelles à l'aide de la Mean Squared Error (MSE).

---

## 3. Exercice 1 — Prévision de la consommation électrique

### 3.1 Objectif

Développer un modèle Transformer capable de prévoir la consommation électrique à partir des observations historiques.

### 3.2 Jeu de données

Le projet utilise le dataset Household Power Consumption.

Les données sont agrégées à une fréquence horaire afin de construire des séquences temporelles.

### 3.3 Préparation des données

Les principales opérations sont :

1. Conversion des informations temporelles.
2. Agrégation horaire des mesures.
3. Traitement des valeurs manquantes.
4. Normalisation à partir des statistiques d'entraînement uniquement.
5. Construction de fenêtres temporelles de 24 heures.
6. Séparation chronologique en ensembles d'entraînement, de validation et de test.

La répartition utilisée est de 70 % pour l'entraînement, 15 % pour la validation et 15 % pour le test.

### 3.4 Architecture du modèle

Le modèle EnergyTransformer utilise les paramètres suivants :

| Paramètre | Valeur |
|---|---:|
| Dimension du modèle | 32 |
| Nombre de têtes d'attention | 4 |
| Nombre de couches | 2 |
| Fenêtre historique | 24 heures |
| Nombre d'époques | 20 |

Le meilleur état du modèle est conservé selon la performance obtenue sur l'ensemble de validation.

### 3.5 Évaluation

Les performances sont mesurées avec deux métriques :

- **MAE** : Mean Absolute Error.
- **RMSE** : Root Mean Squared Error.

Le Transformer est comparé à deux références :

- Persistence : dernière valeur observée.
- Référence 24 heures : valeur observée à la même heure la veille.

Résultats obtenus sur le test :

| Méthode | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| EnergyTransformer | 0.3541 | 0.4918 |
| Persistence | 0.3892 | 0.5884 |
| Référence 24 heures | 0.5133 | 0.7639 |

Le Transformer obtient les meilleures performances parmi les trois méthodes évaluées.

---

## 4. Exercice 2 — Implémentation d'un Mini-GPT

### 4.1 Objectif

Construire un petit modèle de langage autorégressif capable de générer du texte caractère par caractère.

### 4.2 Préparation du corpus

Le corpus textuel est constitué de phrases en français relatives aux modèles de langage et aux Transformers.

Les caractères sont transformés en identifiants numériques à l'aide d'un vocabulaire construit à partir du corpus.

Paramètres principaux :

| Paramètre | Valeur |
|---|---:|
| Taille du vocabulaire | 31 |
| Longueur du contexte | 64 |
| Nombre de blocs GPT | 2 |
| Étapes d'entraînement | 1000 |

### 4.3 Architecture et apprentissage

Le Mini-GPT utilise une architecture Transformer avec attention causale.

Le masque causal empêche le modèle d'accéder aux caractères futurs pendant la prédiction.

L'apprentissage est autorégressif : le modèle apprend à prédire l'élément suivant à partir des éléments précédents.

### 4.4 Génération

La génération est testée avec trois températures :

- 0.5 : génération plus stable.
- 1.0 : génération plus diversifiée.
- 1.5 : génération plus aléatoire.

La comparaison permet d'observer l'influence de la température sur la diversité et la cohérence du texte généré.

---

## 5. Exercice 3 — Implémentation d'un Mini-BERT

### 5.1 Objectif

Développer un modèle de type BERT capable de prédire les mots masqués dans une séquence.

### 5.2 Préparation des données

Un petit corpus de phrases françaises est utilisé.

Les phrases sont transformées en séquences d'identifiants numériques à partir d'un vocabulaire de mots.

Les séquences sont réparties entre entraînement et validation.

### 5.3 Architecture

Le Mini-BERT utilise un encodeur Transformer bidirectionnel.

Paramètres principaux :

| Paramètre | Valeur |
|---|---:|
| Taille du vocabulaire | 51 |
| Nombre de couches | 2 |
| Nombre de têtes d'attention | 4 |
| Nombre d'époques | 30 |

### 5.4 Masked Language Modeling

Pendant l'entraînement, certaines positions sont sélectionnées et leurs mots sont masqués ou corrompus.

Le modèle apprend à retrouver les mots originaux à partir du contexte disponible.

La fonction de perte est calculée uniquement sur les positions sélectionnées.

### 5.5 Évaluation

L'évaluation mesure l'accuracy sur les tokens sélectionnés dans l'ensemble de validation.

Le résultat obtenu est de 100 % sur les positions évaluées.

**Limite expérimentale :** le corpus est très réduit et les phrases sont répétées. Ce résultat ne démontre donc pas une capacité de généralisation à des textes inédits.

---

## 6. Synthèse de la démarche

Les trois exercices illustrent différentes applications des architectures Transformer :

| Modèle | Tâche | Mécanisme principal |
|---|---|---|
| EnergyTransformer | Prévision temporelle | Attention sur les séquences historiques |
| Mini-GPT | Génération de texte | Attention causale |
| Mini-BERT | Prédiction de mots masqués | Attention bidirectionnelle |

Ce travail met en évidence la flexibilité de l'architecture Transformer pour traiter des données séquentielles de différentes natures, qu'il s'agisse de séries temporelles ou de langage naturel.

---

## 7. Limites et pistes d'amélioration

Plusieurs améliorations peuvent être envisagées :

- Tester différentes longueurs de fenêtres temporelles, notamment 168 heures.
- Ajouter des variables temporelles comme l'heure et le jour de la semaine.
- Comparer davantage d'architectures et de configurations.
- Augmenter la taille et la diversité du corpus utilisé pour Mini-GPT et Mini-BERT.
- Évaluer les modèles sur des données indépendantes afin de mieux mesurer leur capacité de généralisation.
- Étudier les performances avec différentes tailles de modèles et différents hyperparamètres.