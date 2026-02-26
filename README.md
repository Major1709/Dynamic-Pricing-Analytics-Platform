# Dynamic Pricing Analytics Platform

Plateforme decisionnelle de pricing dynamique, orientee resultats et impact business.

Cette application permet de transformer des donnees de pricing en decisions actionnables, en combinant:

- visualisation des KPI de performance commerciale
- simulation de scenarios de prix et projection de revenu
- prediction de la demande via un modele ML avec preprocessing
- analyse rapide des leviers de performance (promo, saison, concurrence, time of day)

## Executive Summary

Ce projet regroupe dans une seule interface:

- pilotage business (KPI et dashboards)
- analyse exploratoire (EDA)
- simulation pricing
- prediction de demande

Objectif principal:

- aider les equipes Pricing / Revenue / Data a prendre des decisions plus rapides, plus coherentes et mieux justifiees par la donnee

Resultat attendu:

- passer d'une logique de reporting descriptif a une logique de decision assistee par la simulation et la prediction

Valeur business:

- meilleure visibilite sur les leviers de revenu
- reduction du temps entre analyse et decision
- alignement entre metier et data via des indicateurs communs
- decisions de prix plus defendables (scenarios + metriques + hypotheses explicites)

## Resultats & Impact Business

### Decisions accelerees

- centralise KPI, analyses et simulation dans une interface unique
- reduit les allers-retours entre notebooks, exports CSV et feuilles Excel
- facilite la preparation de supports de decision pour les parties prenantes

### Meilleure qualite de decision prix

- le dashboard ne se limite pas a la visualisation: il integre un modele de prediction pour estimer la demande selon le prix
- le prix "optimal" est derive de simulations (prix -> demande -> revenu), pas uniquement d'une moyenne brute
- les arbitrages prix/volume/revenu deviennent plus explicites

### Gouvernance et reproductibilite

- preprocessing encapsule dans un `Pipeline` sklearn
- metriques de modele tracees (`MAE`, `RMSE`, `R2`)
- artefacts sauvegardables (`.joblib`, `.json`)
- logique de preparation des donnees reutilisable entre training et dashboard

### Exemples de cas d'usage

- evaluer l'effet d'un repositionnement prix avant campagne promo
- comparer la performance par saison / weekday / time of day
- identifier des segments ou la concurrence impacte fortement les ventes
- produire rapidement un support de decision pour equipe business / management

### Mesure de l'impact (recommandee)

Pour piloter l'impact du projet dans un contexte reel, comparer avant/apres sur:

- temps moyen de preparation d'une recommandation pricing
- part des decisions pricing supportees par une simulation ou une prediction
- ecart entre revenu projete et revenu observe
- variation du revenu / marge sur les segments pilotes
- stabilite des performances modele (`RMSE`, `R2`) dans le temps

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

## KPIs a suivre (pilotage des resultats)

Pour une lecture orientee resultat et impact, suivre regulierement:

- revenu projete vs revenu reel (precision business)
- volume vendu apres variation de prix (impact commercial)
- ecart demande predite vs demande observee (qualite predictive)
- evolution de `RMSE` / `R2` apres mise a jour des donnees (sante du modele)
- performance par segment (promo, saison, time of day, weekday) (actionabilite)
- delai entre demande metier et recommandation pricing (efficacite operationnelle)

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

- demos de pricing dynamique orientees valeur
- POC analytics / revenue management
- support de discussion metier-data pour prioriser des experiments pricing
- base de travail pour une application decisionnelle plus industrialisee

## Public cible

- Pricing Manager
- Revenue Manager
- Data Analyst
- Data Scientist / ML Engineer (pour la partie modele et preprocessing)
- Product Owner / Manager (pilotage par KPI et impact)
