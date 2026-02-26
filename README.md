# Dynamic Pricing Analytics Platform

Plateforme Streamlit de pilotage du pricing dynamique, orientee decision et impact business.

Elle permet de:

- transformer des donnees de pricing en decisions actionnables
- estimer la demande en fonction du prix via un modele ML avec preprocessing
- identifier un prix cible optimisant le revenu projete
- analyser rapidement la performance commerciale par saison, promo, moment de la journee et concurrence

## Executive Summary

Ce projet combine visualisation, analyse exploratoire et modelisation predictive dans une meme interface.

Objectif principal:

- aider un responsable pricing / revenue / data a prendre des decisions plus rapides et mieux justifiees

Valeur business:

- meilleure visibilite sur les leviers de revenu
- simulation de scenarios de prix avant mise en production
- standardisation de l'analyse (meme logique de preprocessing, memes metriques)
- reduction du temps entre analyse et decision

## Business Impact (orientation impact)

### Decisions accelerees

- Centralise KPI, analyses et simulation dans une interface unique
- Evite de passer entre notebooks, exports CSV et fichiers Excel

### Meilleure qualite de decision prix

- Le dashboard ne se limite pas a de la visualisation: il integre un modele de prediction pour estimer la demande selon le prix
- Le prix "optimal" est derive de simulations (prix -> demande -> revenu), pas uniquement d'une moyenne brute

### Gouvernance et reproductibilite

- Preprocessing encapsule dans un `Pipeline` sklearn
- Metriques de modele tracees (`MAE`, `RMSE`, `R2`)
- Artefacts sauvegardables (`.joblib`, `.json`)

### Exemples de cas d'usage

- Evaluer l'effet d'un repositionnement prix avant campagne promo
- Comparer performance par saison / weekday / time of day
- Identifier des segments ou la concurrence impacte fortement les ventes
- Produire rapidement un support de decision pour equipe business/management

## Fonctionnalites

### 1) Dashboard de pilotage (`pages/Dashbord.py`)

- KPI metier (demande predite, prix recommande, projection de revenu, elasticite locale)
- courbe d'optimisation prix -> demande -> revenu
- cartes "Metrics Dashboard" (vision management)
- visualisations Plotly (forecast, saisonnalite, concurrence, distribution)
- tableau de simulation de scenarios
- integration directe du modele ML avec preprocessing
- affichage des performances du modele (`RMSE`, `R2`)

### 2) Tableau de donnees (`pages/Tableau.py`)

- recherche multi-colonnes
- pagination
- rendu type DataTable (HTML/CSS)
- formatage monetaire et volumetrique

### 3) Analyse de donnees / EDA (`pages/Analyse_Donnees.py`)

- filtres (date, season, promo, time of day, plage de prix, recherche texte)
- KPIs sur donnees filtrees
- distributions et segmentations
- evolutions temporelles
- statistiques descriptives
- matrice de correlation
- qualite des donnees (valeurs manquantes)
- export CSV du sous-ensemble filtre

### 4) Training modele avec preprocessing (`train_pricing_model.py`)

- preprocessing `ColumnTransformer` + `Pipeline`
- imputation variables numeriques / categorielles
- encodage `OneHotEncoder`
- feature engineering sur `Date`
- controle de fuite de donnees (leakage)
- modele `RandomForestRegressor`
- evaluation (`MAE`, `RMSE`, `R2`)
- sauvegarde modele + metadonnees

## Architecture du projet

```text
Price/
├── app.py                             # Home Streamlit
├── pricing_dataset.csv                # Donnees source
├── train_pricing_model.py             # Training ML + preprocessing + export artefacts
├── artifacts/
│   ├── pricing_demand_model.joblib    # Pipeline sklearn sauvegarde
│   └── pricing_demand_metrics.json    # Metriques / metadata du modele
└── pages/
    ├── Dashbord.py                    # Dashboard principal (visualisation + modele)
    ├── Tableau.py                     # Tableau pagine / recherche
    └── Analyse_Donnees.py             # Analyse exploratoire (EDA)
```

## Stack technique

- Python 3.10+
- Streamlit
- pandas / numpy
- Plotly
- scikit-learn
- joblib

## Installation

### 1. Creer un environnement virtuel

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

### Lancer la plateforme

```bash
streamlit run app.py
```

Pages disponibles dans la sidebar:

- `Dashbord`
- `Tableau`
- `Analyse_Donnees`

## Entrainement du modele (CLI)

### Entrainement + sauvegarde (par defaut)

```bash
python3 train_pricing_model.py
```

### Test rapide sans sauvegarde

```bash
python3 train_pricing_model.py --no-save
```

### Exemple avec parametres explicites

```bash
python3 train_pricing_model.py \
  --data-path pricing_dataset.csv \
  --target "Units Sold" \
  --test-size 0.2 \
  --model-out artifacts/pricing_demand_model.joblib \
  --metrics-out artifacts/pricing_demand_metrics.json
```

## Integration du modele dans le dashboard

Le dashboard principal reutilise la logique du script de training pour garantir la coherence entre:

- preprocessing applique au modele
- predictions de simulation affichees dans l'interface
- metriques de performance exposees au metier

Concretement, le dashboard:

- entraine un pipeline (cache Streamlit)
- simule la demande pour une grille de prix
- calcule un prix maximisant le revenu projete
- affiche les KPI et scenarios issus de ce modele

Notes d'exploitation:

- premier chargement potentiellement plus lent (entrainement)
- reruns acceleres via `@st.cache_resource`
- recalcul automatique si `pricing_dataset.csv` est modifie

## Schema de donnees attendu

Colonnes attendues dans `pricing_dataset.csv`:

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

Lors d'un entrainement avec sauvegarde:

- `artifacts/pricing_demand_model.joblib`
- `artifacts/pricing_demand_metrics.json`

Le fichier de metriques contient notamment:

- metriques d'evaluation (`mae`, `rmse`, `r2`)
- tailles train/test
- metadata de preprocessing (features numeriques/categorielles)
- configuration du modele / split

## KPIs a suivre (suggestion)

Pour un usage "impact business", suivre regulierement:

- revenu projete vs revenu reel
- taux de conversion / volume de ventes apres changement de prix
- ecart entre demande predite et demande observee
- evolution de `RMSE` / `R2` apres mise a jour des donnees
- performance par segment (promo, saison, time of day)

## Troubleshooting

### Texte peu visible / conflit de theme Streamlit

Le projet applique des styles CSS sur fond clair.
En cas de rendu incoherent:

- faire `Ctrl+F5`
- relancer Streamlit

Optionnel: forcer un theme clair via `.streamlit/config.toml`

```toml
[theme]
base = "light"
```

### Warning joblib (serial mode)

Un warning de type `joblib will operate in serial mode` peut apparaitre selon l'environnement.
Ce warning n'empeche pas l'entrainement ni l'inference.

### Aucun resultat apres filtrage (EDA)

Verifier:

- periode selectionnee
- filtres (`Season`, `Promo`, `Time of Day`)
- plage de prix
- recherche texte

## Commandes utiles

```bash
# Lancer l'application
streamlit run app.py

# Entrainer le modele (avec sauvegarde)
python3 train_pricing_model.py

# Tester le pipeline sans sauvegarde
python3 train_pricing_model.py --no-save
```

## Roadmap / ameliorations

- page de prediction dediee (formulaire d'inference)
- bouton "Reentrainer le modele" dans le dashboard
- chargement du modele depuis `artifacts/` au lieu de re-entrainer dans la page
- export PDF / image des dashboards
- `requirements.txt` ou `pyproject.toml`
- tests unitaires (preprocessing, training, inference)
- suivi des performances modele dans le temps (monitoring)

## Positionnement

Ce projet est adapte a:

- demos de pricing dynamique
- POC analytics / revenue management
- base de travail pour une application decisionnelle plus industrialisee

