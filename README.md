---
noteId: "c48df4e0d60a11f0812c4bc3ea7e502e"
tags: []

---

# Laboratoires de Machine Learning et Deep Learning

**Auteur:** FILALI ANSARI Meryem  

**Date:** Décembre 2025

---

## Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Laboratoire 1 : Régression Linéaire](#laboratoire-1--régression-linéaire)
3. [Laboratoire 2 : Classification et Réseaux de Neurones](#laboratoire-2--classification-et-réseaux-de-neurones)
4. [Dépendances et Installation](#dépendances-et-installation)
5. [Structure des Fichiers](#structure-des-fichiers)
6. [Références](#références)

---

## Vue d'Ensemble

Ce repository contient deux laboratoires pratiques explorant les concepts fondamentaux du Machine Learning et du Deep Learning :

- **Laboratoire 1** : Implémentation et comparaison de la régression linéaire avec trois approches différentes
- **Laboratoire 2** : Classification binaire avec comparaison des fonctions d'activation et architectures de réseaux de neurones

Les notebooks démontrent une progression pédagogique depuis l'implémentation manuelle des algorithmes jusqu'à l'utilisation de frameworks modernes comme TensorFlow.

---

## Laboratoire 1 : Régression Linéaire

### Objectif

Prédire le prix d'un bien immobilier en fonction de sa surface en mètres carrés, en implémentant la régression linéaire de trois façons différentes pour comprendre les fondements mathématiques et algorithmiques.

### Dataset

**Caractéristiques du dataset synthétique :**
- Nombre d'échantillons : 100
- Variable indépendante (X) : Surface en m² (valeurs entre 30 et 200 m²)
- Variable dépendante (y) : Prix en euros
- Relation : Prix = 2000 × Surface + Bruit gaussien (μ=0, σ=20000)
- Division : 80% entraînement (80 échantillons), 20% test (20 échantillons)

### Méthodologies Implémentées

#### 1. Implémentation Manuelle - Gradient Descent

**Principe mathématique :**

La régression linéaire cherche à trouver les paramètres θ₀ (intercept) et θ₁ (pente) qui minimisent la fonction de coût Mean Squared Error (MSE) :

```
J(θ₀, θ₁) = (1/2m) × Σ(ŷᵢ - yᵢ)²
où ŷᵢ = θ₀ + θ₁ × xᵢ
```

**Algorithme de descente de gradient :**

```
Initialisation : θ₀ = 0, θ₁ = 0
Pour chaque itération :
    1. Calcul des prédictions : ŷ = θ₀ + θ₁ × X
    2. Calcul des gradients :
       ∂J/∂θ₀ = (1/m) × Σ(ŷᵢ - yᵢ)
       ∂J/∂θ₁ = (1/m) × Σ((ŷᵢ - yᵢ) × xᵢ)
    3. Mise à jour des paramètres :
       θ₀ = θ₀ - α × ∂J/∂θ₀
       θ₁ = θ₁ - α × ∂J/∂θ₁
```

**Paramètres utilisés :**
- Learning rate (α) : 0.01
- Nombre d'itérations : 1000
- Normalisation : Z-score (X_norm = (X - μ) / σ)

**Résultats obtenus :**
- Intercept (θ₀) : ~0 € (après dénormalisation)
- Pente (θ₁) : ~2000 €/m²
- Convergence : Loss diminue progressivement jusqu'à stabilisation
- Performance : MSE et R² comparables aux méthodes optimisées

**Visualisations générées :**
1. Convergence de la fonction de perte (Loss) sur 1000 itérations
2. Évolution de θ₀ (biais) au fil des itérations
3. Évolution de θ₁ (pente) au fil des itérations
4. Modèle ajusté sur données d'entraînement
5. Modèle testé sur données de test
6. Trajectoire 2D dans l'espace des paramètres (θ₀, θ₁)

#### 2. Scikit-Learn

**Approche :**

Utilisation de la classe `LinearRegression` de scikit-learn qui implémente une solution analytique basée sur les moindres carrés ordinaires (Ordinary Least Squares - OLS).

**Solution analytique :**

```
θ = (XᵀX)⁻¹Xᵀy
```

Cette méthode calcule directement les paramètres optimaux sans itérations, ce qui est plus rapide pour les petits datasets.

**Code principal :**

```python
from sklearn.linear_model import LinearRegression

model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred = model_sklearn.predict(X_test)
```

**Avantages :**
- Calcul direct sans hyperparamètres à ajuster
- Solution exacte garantie
- Implémentation optimisée en C
- API simple et intuitive

**Résultats :**
- Intercept : identique à la méthode manuelle
- Coefficient : identique à la méthode manuelle
- Temps d'exécution : quasi-instantané
- MSE et R² : performance optimale

#### 3. TensorFlow / Keras

**Approche :**

Construction d'un réseau de neurones minimal avec une seule couche Dense de 1 neurone et activation linéaire, équivalent à une régression linéaire mais entraîné par rétropropagation.

**Architecture du modèle :**

```
Input (1 feature) → Dense(1, activation='linear') → Output
```

**Configuration :**
- Optimizer : SGD (Stochastic Gradient Descent) avec learning rate = 0.01
- Loss : MSE (Mean Squared Error)
- Epochs : 1000
- Normalisation : Z-score appliquée avant entraînement

**Code principal :**

```python
import tensorflow as tf
from tensorflow import keras

model_tf = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,), activation='linear')
])

model_tf.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

history = model_tf.fit(X_train_norm, y_train_norm, 
                       epochs=1000, 
                       validation_split=0.2,
                       verbose=0)
```

**Avantages :**
- Extensible à des architectures plus complexes
- Support du GPU pour grands datasets
- Historique d'entraînement accessible
- Framework industriel standard

**Visualisations TensorFlow :**
1. Courbe de convergence (Loss sur train et validation)
2. Prédictions sur ensemble d'entraînement
3. Prédictions sur ensemble de test

### Comparaison des Trois Méthodes

| Critère | Gradient Descent Manuel | Scikit-Learn | TensorFlow |
|---------|------------------------|--------------|------------|
| **Intercept** | ~0 € | ~0 € | ~0 € |
| **Pente** | ~2000 €/m² | ~2000 €/m² | ~2000 €/m² |
| **MSE** | ~400,000,000 | ~400,000,000 | ~400,000,000 |
| **R² Score** | ~0.95 | ~0.95 | ~0.95 |
| **Temps calcul** | ~100 ms | <10 ms | ~500 ms |
| **Complexité** | Haute | Faible | Moyenne |
| **Flexibilité** | Maximale | Limitée | Très haute |
| **Usage pédagogique** | Excellent | Bon | Excellent |

### Métriques d'Évaluation

**Mean Squared Error (MSE) :**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```
Mesure la moyenne des erreurs au carré. Plus la valeur est faible, meilleur est le modèle.

**R² Score (Coefficient de détermination) :**
```
R² = 1 - (SS_res / SS_tot)
où SS_res = Σ(yᵢ - ŷᵢ)²  (somme des carrés des résidus)
    SS_tot = Σ(yᵢ - ȳ)²  (somme totale des carrés)
```
Varie entre 0 et 1. Une valeur de 0.95 signifie que 95% de la variance est expliquée par le modèle.

### Prédictions sur Nouvelles Données

Le notebook démontre l'application des trois modèles entraînés sur de nouvelles surfaces :

| Surface (m²) | Prix Manuel | Prix Sklearn | Prix TensorFlow |
|--------------|-------------|--------------|-----------------|
| 75 | ~150,000 € | ~150,000 € | ~150,000 € |
| 120 | ~240,000 € | ~240,000 € | ~240,000 € |
| 150 | ~300,000 € | ~300,000 € | ~300,000 € |

Les trois méthodes produisent des prédictions quasi-identiques, validant leur équivalence mathématique.

### Points Clés du Laboratoire 1

- **Compréhension profonde** : Implémentation from scratch du gradient descent
- **Normalisation essentielle** : Améliore la convergence et la stabilité numérique
- **Équivalence des méthodes** : Trois approches différentes, mêmes résultats
- **Trade-offs** : Vitesse vs. flexibilité vs. compréhension
- **Validation** : Importance de la séparation train/test

---

## Laboratoire 2 : Classification et Réseaux de Neurones

### Objectif

Explorer la classification binaire avec différentes architectures de réseaux de neurones et comparer l'impact des fonctions d'activation (ReLU vs Sigmoid) sur des datasets de complexité variable.

### Datasets Utilisés

#### 1. Dataset Synthétique avec Séparation Nette (+/-2)

**Génération :**
```python
X_class_0 = np.random.randn(250, 2) + np.array([-2, -2])
X_class_1 = np.random.randn(250, 2) + np.array([2, 2])
```

**Caractéristiques :**
- 500 échantillons (250 par classe)
- 2 features (Feature 1, Feature 2)
- Classes bien séparées dans l'espace 2D
- Séparation linéaire possible

**Objectif :** Vérifier que les modèles simples fonctionnent bien sur des données linéairement séparables.

#### 2. Dataset Synthétique avec Écart Minimal (+/-0.0001)

**Génération :**
```python
X_class_0 = np.random.randn(250, 2) + np.array([-0.0001, -0.0001])
X_class_1 = np.random.randn(250, 2) + np.array([0.0001, 0.0001])
```

**Caractéristiques :**
- Classes quasi-superposées
- Problème de classification extrêmement difficile
- Nécessite apprentissage de patterns subtils

**Objectif :** Tester la capacité des réseaux de neurones à extraire des features discriminantes même dans des cas limites.

#### 3. Make_Moons (Scikit-Learn)

**Génération :**
```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
```

**Caractéristiques :**
- 200 échantillons
- Distribution en forme de croissants de lune imbriqués
- Non-linéairement séparable
- Benchmark classique pour tester les classifieurs non-linéaires

**Objectif :** Démontrer la supériorité des réseaux de neurones sur des données complexes.

### Modèles et Architectures

#### 1. Régression Logistique (Baseline)

**Implémentation Scikit-Learn :**

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
```

**Principe mathématique :**

La régression logistique applique la fonction sigmoïde à une combinaison linéaire des features :

```
P(y=1|X) = σ(θ₀ + θ₁x₁ + θ₂x₂)
où σ(z) = 1 / (1 + e⁻ᶻ)
```

**Résultats :**
- Dataset séparé (+/-2) : Accuracy ~100%
- Make_moons : Accuracy ~86-88%

**Conclusion :** Excellent sur données linéaires, limité sur données non-linéaires.

#### 2. Régression Logistique avec TensorFlow

**Architecture :**
```python
model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(2,))
])
```

**Configuration :**
- 1 seul neurone (équivalent à régression logistique)
- Activation : Sigmoid
- Loss : Binary Crossentropy
- Optimizer : Adam

**Résultats :**
- Performances similaires à Scikit-Learn
- Historique d'entraînement accessible
- Base pour extensions futures

#### 3. Réseaux de Neurones avec ReLU

**Architecture :**
```python
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

**Fonction ReLU (Rectified Linear Unit) :**
```
ReLU(x) = max(0, x) = {
    x  si x > 0
    0  si x ≤ 0
}
```

**Propriétés de ReLU :**
- Non-linéaire mais calcul simple
- Pas de problème de gradient qui disparaît (vanishing gradient)
- Sparse activation (beaucoup de neurones à 0)
- Standard dans les réseaux profonds modernes

**Résultats :**
- Dataset écart minimal : Accuracy ~50-60%
- Make_moons : Accuracy ~95-97%

**Analyse :** ReLU excelle sur données structurées non-linéaires.

#### 4. Réseaux de Neurones avec Sigmoid

**Architecture :**
```python
model = keras.Sequential([
    layers.Dense(16, activation='sigmoid', input_shape=(2,)),
    layers.Dense(8, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
])
```

**Fonction Sigmoid :**
```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Propriétés de Sigmoid :**
- Sortie bornée entre 0 et 1
- Interprétable comme probabilité
- Problème de vanishing gradient sur réseaux profonds
- Convergence plus lente que ReLU

**Résultats :**
- Dataset écart minimal : Accuracy ~50-55%
- Make_moons : Accuracy ~85-90%

**Analyse :** Performance inférieure à ReLU sur données complexes, surtout avec architectures profondes.

#### 5. Réseau de Neurones Complet (Deep Neural Network)

**Architecture finale :**
```python
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(2,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

**Configuration d'entraînement :**
- Optimizer : Adam (Adaptive Moment Estimation)
- Loss : Binary Crossentropy
- Epochs : 100
- Batch size : 32
- Validation split : 20%

**Résultats :**
- Dataset séparé : Accuracy ~100%
- Make_moons : Accuracy ~97-99%

**Analyse :** Architecture profonde capture efficacement les patterns complexes.

### Comparaison ReLU vs Sigmoid

#### Courbes d'Apprentissage

**Observations sur Make_Moons :**

1. **ReLU :**
   - Convergence rapide (20-30 epochs)
   - Accuracy finale : 95-97%
   - Loss finale : ~0.10-0.15
   - Pas de surapprentissage significatif

2. **Sigmoid :**
   - Convergence plus lente (50-70 epochs)
   - Accuracy finale : 85-90%
   - Loss finale : ~0.30-0.35
   - Écart train/validation plus grand

#### Frontières de Décision

Les visualisations montrent que :
- **ReLU** : Frontières nettes et bien adaptées aux formes complexes
- **Sigmoid** : Frontières plus lisses mais moins précises
- **Régression Logistique** : Frontière linéaire inadaptée aux croissants

### Terminologie des Réseaux de Neurones

Le laboratoire inclut une section pédagogique détaillant :

- **Neurone** : Unité de base calculant une somme pondérée + activation
- **Couche d'entrée (Input Layer)** : Reçoit les features
- **Couches cachées (Hidden Layers)** : Transforment les données
- **Couche de sortie (Output Layer)** : Produit la prédiction finale
- **Poids (Weights)** : Paramètres appris pendant l'entraînement
- **Biais (Bias)** : Terme constant ajouté à chaque neurone
- **Forward propagation** : Calcul de la prédiction
- **Backpropagation** : Calcul des gradients pour ajuster les poids
- **Epoch** : Une passe complète sur toutes les données
- **Batch** : Sous-ensemble de données traité en une fois
- **Loss** : Mesure de l'erreur
- **Optimizer** : Algorithme d'ajustement des poids

### Matrices de Confusion et Classification Report

**Exemple sur Make_Moons avec NN Complet :**

```
              precision    recall  f1-score   support

   Classe 0       0.98      0.97      0.98        20
   Classe 1       0.97      0.98      0.98        20

   accuracy                           0.98        40
  macro avg       0.98      0.98      0.98        40
weighted avg      0.98      0.98      0.98        40
```

**Interprétation :**
- Precision : Proportion de prédictions positives correctes
- Recall : Proportion d'exemples positifs correctement identifiés
- F1-score : Moyenne harmonique de precision et recall
- Support : Nombre d'échantillons par classe

### Résultats Comparatifs Finaux

| Modèle | Dataset | Accuracy | Loss Finale | Temps Entraînement |
|--------|---------|----------|-------------|-------------------|
| Régression Logistique (Sklearn) | Séparé (+/-2) | 1.0000 | - | <10 ms |
| Régression Linéaire (Sklearn) | Séparé (+/-2) | 1.0000 | - | <10 ms |
| Régression Logistique (TF) | Séparé (+/-2) | 0.9900 | 0.05 | ~200 ms |
| NN avec ReLU | Écart minimal | 0.5800 | 0.68 | ~3 s |
| NN avec Sigmoid | Écart minimal | 0.5200 | 0.70 | ~3 s |
| NN avec ReLU | Make_moons | 0.9700 | 0.12 | ~3 s |
| NN avec Sigmoid | Make_moons | 0.8800 | 0.32 | ~3 s |
| NN Complet (séparé) | Séparé (+/-2) | 1.0000 | 0.01 | ~5 s |
| NN Complet (moons) | Make_moons | 0.9850 | 0.08 | ~5 s |

### Visualisations Générées

Le laboratoire 2 produit :

1. **Scatter plots** : Distribution des classes dans l'espace 2D
2. **Frontières de décision** : Zones de classification par chaque modèle
3. **Courbes d'apprentissage** : Évolution de l'accuracy et de la loss
4. **Comparaisons ReLU vs Sigmoid** : Performance côte à côte
5. **Matrices de confusion** : Analyse détaillée des erreurs
6. **Graphiques en barres** : Comparaisons agrégées par dataset et activation

### Points Clés du Laboratoire 2

- **Fonctions d'activation** : ReLU supérieure à Sigmoid pour la plupart des cas
- **Architecture** : Plus de couches = meilleure capacité d'apprentissage
- **Non-linéarité** : Essentielle pour données complexes (make_moons)
- **Régularisation** : Validation split pour éviter le surapprentissage
- **Optimizers** : Adam converge plus vite que SGD
- **Métriques multiples** : Accuracy, Loss, Precision, Recall pour analyse complète

---

## Dépendances et Installation

### Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
statsmodels>=0.13.0
```

### Installation

**Méthode 1 : pip**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels
```

**Méthode 2 : conda**

```bash
conda install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels
```

**Méthode 3 : requirements.txt**

```bash
pip install -r requirements.txt
```

### Vérification de l'Installation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
import tensorflow as tf

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"TensorFlow version: {tf.__version__}")
```

### Environnement Recommandé

- **Python** : 3.8 ou supérieur
- **RAM** : Minimum 4 GB (8 GB recommandé)
- **Système** : Windows 10/11, Linux, macOS
- **IDE** : Jupyter Notebook, JupyterLab, VS Code avec extension Python

---

## Structure des Fichiers

```
mssror/
├── Livrable_1_FILALI_ANSARI_Meryem.ipynb    # Lab 1 : Régression Linéaire
├── Livrable_2_FILALI_ANSARI_MERYEM.ipynb    # Lab 2 : Classification et NN
├── Time_series_LAB_FILALI_ANSARI_MERYEM.ipynb  # Lab séries temporelles
├── README_LABS.md                            # Ce fichier
├── README.md                                 # README séries temporelles
├── .git/                                     # Version control
└── images/                                   # (optionnel) Graphiques exportés
```

---

## Exécution des Notebooks

### Jupyter Notebook

```bash
cd C:\Users\awati\Desktop\mssror
jupyter notebook
```

Puis ouvrir :
- `Livrable_1_FILALI_ANSARI_Meryem.ipynb`
- `Livrable_2_FILALI_ANSARI_MERYEM.ipynb`

### VS Code

1. Ouvrir VS Code dans le répertoire
2. Installer l'extension Python et Jupyter
3. Ouvrir le fichier .ipynb
4. Sélectionner le kernel Python approprié
5. Exécuter les cellules séquentiellement

### Google Colab

1. Aller sur [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook
3. Sélectionner le fichier .ipynb
4. Exécuter les cellules

---

## Concepts Théoriques Abordés

### Machine Learning

- **Apprentissage supervisé** : Régression et classification
- **Séparation train/test** : Validation de la généralisation
- **Fonction de coût** : MSE, Binary Crossentropy
- **Optimisation** : Gradient Descent, Adam
- **Métriques** : MSE, R², Accuracy, Precision, Recall, F1-Score
- **Normalisation** : Z-score, Min-Max
- **Overfitting vs Underfitting** : Compromis biais-variance

### Deep Learning

- **Réseaux de neurones** : Architecture multi-couches
- **Fonctions d'activation** : ReLU, Sigmoid, Linear
- **Rétropropagation** : Algorithme de calcul des gradients
- **Optimizers** : SGD, Adam, RMSprop
- **Régularisation** : Validation split, Dropout (non utilisé ici)
- **Convergence** : Courbes d'apprentissage
- **Frontières de décision** : Visualisation des zones de classification

### Mathématiques

- **Algèbre linéaire** : Matrices, produits matriciels
- **Calcul différentiel** : Dérivées partielles, gradients
- **Probabilités** : Distribution gaussienne, fonction sigmoïde
- **Statistiques** : Moyenne, variance, écart-type
- **Optimisation** : Descente de gradient, minima locaux/globaux

---

## Résultats et Conclusions

### Laboratoire 1

**Conclusion principale :** Les trois méthodes (manuelle, Scikit-Learn, TensorFlow) produisent des résultats équivalents pour la régression linéaire, validant :
1. La robustesse de l'algorithme de gradient descent
2. L'importance de la normalisation
3. La flexibilité de TensorFlow pour des architectures simples et complexes

**Application pratique :** Le modèle entraîné peut prédire avec précision le prix d'un bien immobilier, avec R² ~ 0.95, indiquant que 95% de la variance du prix est expliquée par la surface.

### Laboratoire 2

**Conclusion principale :** Les réseaux de neurones avec activation ReLU surpassent significativement les méthodes linéaires et l'activation Sigmoid sur des données non-linéaires :
1. **Données linéaires** : Tous les modèles performent bien (accuracy ~100%)
2. **Données non-linéaires (make_moons)** : ReLU atteint 97% vs 88% pour Sigmoid vs 86% pour régression logistique
3. **Architecture profonde** : Améliore la capacité d'apprentissage

**Application pratique :** Pour des problèmes de classification complexes, privilégier :
- Architectures profondes (3-4 couches cachées)
- Activation ReLU dans les couches cachées
- Activation Sigmoid en sortie pour classification binaire
- Optimizer Adam pour convergence rapide

---

## Perspectives et Extensions

### Pour le Laboratoire 1

1. **Régression polynomiale** : Tester des relations non-linéaires
2. **Régularisation** : Implémentation de Ridge (L2) et Lasso (L1)
3. **Features multiples** : Ajouter nombre de chambres, localisation, etc.
4. **Cross-validation** : K-fold pour validation plus robuste
5. **Détection d'outliers** : Influence sur les prédictions

### Pour le Laboratoire 2

1. **Autres activations** : Tanh, Leaky ReLU, ELU, Swish
2. **Régularisation** : Dropout, L2 regularization, Batch Normalization
3. **Optimizers** : Comparaison SGD vs Adam vs RMSprop vs AdaGrad
4. **Hyperparameter tuning** : Grid search, Random search
5. **Classification multi-classes** : Extension à 3+ classes
6. **Architectures avancées** : ResNet, DenseNet pour images
7. **Transfer learning** : Utilisation de modèles pré-entraînés

---

## Références

### Livres

1. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** (3rd Edition)  
   Aurélien Géron (2022)  
   O'Reilly Media

2. **Deep Learning**  
   Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)  
   MIT Press

3. **Pattern Recognition and Machine Learning**  
   Christopher M. Bishop (2006)  
   Springer

### Articles Académiques

1. **Gradient-Based Learning Applied to Document Recognition**  
   Y. LeCun et al. (1998)  
   Proceedings of the IEEE

2. **Rectified Linear Units Improve Restricted Boltzmann Machines**  
   Vinod Nair, Geoffrey E. Hinton (2010)  
   ICML

3. **Adam: A Method for Stochastic Optimization**  
   Diederik P. Kingma, Jimmy Ba (2014)  
   ICLR

### Documentation en Ligne

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### Tutoriels

- [Machine Learning Crash Course - Google](https://developers.google.com/machine-learning/crash-course)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

---

## Auteur et Contact

**FILALI ANSARI Meryem**

Étudiante en Machine Learning et Intelligence Artificielle  
Année académique 2024-2025

**Repository GitHub :** [LAB_TIME_SERIES](https://github.com/meryemfilaliansari/LAB_TIME_SERIES)

---

## Licence

Ce projet est à usage éducatif et pédagogique dans le cadre universitaire.

---

## Changelog

**Version 1.0 (Décembre 2025)**
- Laboratoire 1 : Régression linéaire complète avec 3 implémentations
- Laboratoire 2 : Classification binaire avec comparaison ReLU vs Sigmoid
- Documentation complète et détaillée
- Visualisations et analyses comparatives

---

**Dernière mise à jour :** Décembre 2025  
**Statut :** Complet et fonctionnel  
**Version Python :** 3.11+  
**Version TensorFlow :** 2.15+
