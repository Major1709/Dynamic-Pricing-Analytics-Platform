# Dynamic Pricing Analytics Platform

Application Streamlit de pricing dynamique pour:

- visualiser les indicateurs metier (KPI, projections, simulations)
- explorer le dataset de pricing (tableau + EDA)
- entrainer un modele de prediction avec preprocessing sklearn
- integrer les predictions du modele directement dans le dashboard

Le projet s'appuie sur `pricing_dataset.csv` comme source de donnees principale.

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Fonctionnalites](#fonctionnalites)
- [Stack technique](#stack-technique)
- [Installation](#installation)
- [Execution](#execution)
- [Entrainement du modele (CLI)](#entrainement-du-modele-cli)
- [Integration du modele dans le dashboard](#integration-du-modele-dans-le-dashboard)
- [Schema de donnees attendu](#schema-de-donnees-attendu)
- [Artefacts generes](#artefacts-generes)
- [Troubleshooting](#troubleshooting)
- [Roadmap / ameliorations](#roadmap--ameliorations)

## Vue d'ensemble

Ce projet fournit une interface decisionnelle pour un cas de pricing dynamique.
Il combine:

- un dashboard de pilotage (`pages/Dashbord.py`)
- un tableau de consultation de donnees (`pages/Tableau.py`)
- une page d'analyse exploratoire (`pages/Analyse_Donnees.py`)
- un pipeline de machine learning avec preprocessing (`train_pricing_model.py`)

Le dashboard utilise un modele de prediction integre pour estimer la demande en fonction du prix et calculer des scenarios de revenus.

## Architecture du projet

```text
Price/
├── app.py                             # Page d'accueil Streamlit
├── pricing_dataset.csv                # Donnees source
├── train_pricing_model.py             # Training + preprocessing + export des artefacts
├── artifacts/
│   ├── pricing_demand_model.joblib    # Modele sauvegarde (pipeline sklearn)
│   └── pricing_demand_metrics.json    # Metriques et metadata du modele
└── pages/
    ├── Dashbord.py                    # Dashboard principal (KPI + visualisations + simulation)
    ├── Tableau.py                     # Tableau pagine / recherche multi-colonnes
    └── Analyse_Donnees.py             # EDA / statistiques / correlations / export CSV
```

## Fonctionnalites

### Dashboard (`pages/Dashbord.py`)

- KPI metier (demande predite, prix optimal, projection de revenu, elasticite locale)
- courbe d'optimisation prix -> demande -> revenu
- visualisations Plotly (forecast, saisonnalite, concurrence, distribution, etc.)
- simulation de scenarios de pricing
- cartes de performance du modele (RMSE / R2)
- integration du pipeline de machine learning avec preprocessing (cache Streamlit)

### Tableau (`pages/Tableau.py`)

- recherche texte sur toutes les colonnes
- pagination
- rendu HTML stylise (inspire DataTable)
- formatage des montants et volumes

### Analyse de donnees (`pages/Analyse_Donnees.py`)

- filtres interactifs (date, saison, promo, time of day, plage de prix, recherche)
- KPI sur sous-ensemble filtre
- distributions, evolutions et segmentations
- statistiques descriptives
- matrice de correlation
- controle des valeurs manquantes
- export CSV des donnees filtrees

### Training du modele (`train_pricing_model.py`)

- preprocessing via `ColumnTransformer` + `Pipeline`
- imputation des variables numeriques et categorielles
- encodage `OneHotEncoder`
- feature engineering sur `Date`
- gestion de la fuite de donnees (ex: retrait de `Revenue ($)` si prediction de `Units Sold`)
- entrainement `RandomForestRegressor`
- evaluation (`MAE`, `RMSE`, `R2`)
- sauvegarde du pipeline et des metriques

## Stack technique

- Python 3.10+
- Streamlit
- pandas / numpy
- Plotly
- scikit-learn
- joblib

## Installation

### 1. Creer et activer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dependances

```bash
pip install --upgrade pip
pip install streamlit pandas numpy plotly scikit-learn joblib
```

## Execution

### Lancer l'application Streamlit

```bash
streamlit run app.py
```

Ensuite, naviguer dans la sidebar Streamlit vers:

- `Dashbord`
- `Tableau`
- `Analyse_Donnees`

## Entrainement du modele (CLI)

### Entrainement + sauvegarde (par defaut)

```bash
python3 train_pricing_model.py
```

### Entrainement sans sauvegarde (verification rapide)

```bash
python3 train_pricing_model.py --no-save
```

### Parametres disponibles (exemples)

```bash
python3 train_pricing_model.py \
  --data-path pricing_dataset.csv \
  --target "Units Sold" \
  --test-size 0.2 \
  --model-out artifacts/pricing_demand_model.joblib \
  --metrics-out artifacts/pricing_demand_metrics.json
```

## Integration du modele dans le dashboard

Le dashboard principal utilise des fonctions de `train_pricing_model.py` pour:

- preparer les features avec preprocessing identique au training
- entrainer (ou recalculer) un pipeline sur les donnees courantes
- generer des predictions "what-if" en fonction du prix
- calculer le prix optimal (maximisation du revenu projete)
- afficher les metriques du modele (RMSE / R2)

Notes d'exploitation:

- le premier chargement de `pages/Dashbord.py` peut etre plus lent (entrainement)
- les executions suivantes sont plus rapides grace a `@st.cache_resource`
- si `pricing_dataset.csv` change, le cache est rafraichi

## Schema de donnees attendu

Le dataset `pricing_dataset.csv` est attendu avec les colonnes suivantes:

- `Date`
- `Price ($)`
- `Units Sold`
- `Promo`
- `Weekday`
- `Competitor Price ($)`
- `Season`
- `Time of Day`
- `Revenue ($)`

## Artefacts generes

Lors de l'entrainement (sans `--no-save`), les fichiers suivants sont generes:

- `artifacts/pricing_demand_model.joblib`
- `artifacts/pricing_demand_metrics.json`

Contenu typique de `pricing_demand_metrics.json`:

- metriques d'evaluation (`mae`, `rmse`, `r2`)
- volumetrie (train/test)
- metadata preprocessing (features numeriques/categorielles)
- type de modele et configuration de split

## Troubleshooting

### 1. Texte peu visible (theme sombre / CSS)

Le projet applique du CSS pour ameliorer la lisibilite sur cartes blanches.
Si le rendu n'est pas a jour:

- recharger la page avec `Ctrl+F5`
- relancer Streamlit

Optionnel: forcer un theme clair via `.streamlit/config.toml`

```toml
[theme]
base = "light"
```

### 2. Warning joblib (mode serial)

Dans certains environnements, un warning peut apparaitre:

- `joblib will operate in serial mode`

Ce warning n'empeche pas l'entrainement ni l'utilisation du modele.

### 3. Aucune donnee apres filtrage (page Analyse_Donnees)

Verifier:

- plage de dates
- filtres `Season`, `Promo`, `Time of Day`
- plage de prix
- champ de recherche texte

## Commandes utiles

```bash
# Lancer l'application
streamlit run app.py

# Entrainer le modele et sauvegarder les artefacts
python3 train_pricing_model.py

# Tester le pipeline sans sauvegarde
python3 train_pricing_model.py --no-save
```

## Roadmap / ameliorations

- page de prediction dediee (formulaire d'inference)
- bouton "Reentrainer le modele" dans le dashboard
- chargement direct du modele depuis `artifacts/` (si disponible)
- export PDF / image des dashboards
- `requirements.txt` et/ou `pyproject.toml`
- tests unitaires (preprocessing, training, inference)
- monitoring des metriques du modele dans le temps

## Auteur / contexte

Projet de demonstration/analysis pour un cas de pricing dynamique avec Streamlit + scikit-learn.

