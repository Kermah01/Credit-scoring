# Credit Scoring App — Démo

Application web de **scoring de crédit bancaire** développée avec Streamlit :
elle estime la probabilité de défaut de paiement d'un emprunteur à partir d'un
modèle de machine learning, et propose un tableau de bord interactif d'analyse
du portefeuille de prêts.

> ⚠️ **Données 100 % fictives.** L'intégralité des données (clients, agences,
> prêts, localisations) est générée synthétiquement, avec une graine fixe, à des
> fins de démonstration. Aucun client réel ni établissement bancaire réel n'est
> représenté, et les scores produits n'ont aucune valeur de décision de crédit.

## Fonctionnalités

- **Scoring interactif** : saisie d'un dossier de prêt (montant, durée, taux,
  profil de l'emprunteur, garanties) et calcul en temps réel de la probabilité
  de défaut, restituée via une jauge Plotly et une synthèse chiffrée.
- **Tableau de bord** du portefeuille : indicateurs clés (volume, montant moyen,
  taux de défaut), analyses univariées et croisées, distributions, production de
  crédits dans le temps, et carte interactive du risque par agence.
- **Filtres dynamiques** par agence, type de prêt, période d'octroi et statut.
- **Pipeline ML reproductible** : génération des données synthétiques,
  entraînement (régression logistique vs forêt aléatoire, validation croisée)
  et sérialisation du meilleur modèle sous forme de Pipeline scikit-learn
  complet (prétraitement inclus).

## Aperçu

```
┌─────────────────────────────────────────────────────────────┐
│  💳 Credit Scoring App — Démo                               │
│  ⚠️ Démonstration — données 100 % fictives                  │
├──────────────┬──────────────────────────────────────────────┤
│  Dossier     │   Probabilité de défaut        Synthèse      │
│  ─ Montant   │        ╭─────────╮         ┌──────┬──────┐   │
│  ─ Durée     │        │  23,4 % │         │ 23,4%│ 5 M  │   │
│  ─ Taux      │        ╰─────────╯         ├──────┼──────┤   │
│  ─ Profil    │     🟢──🟡──🟠──🔴          │ 48 m │ 8,5 %│   │
│  ─ Garanties │                            └──────┴──────┘   │
│  [Calculer]  │   ✅ Risque faible — profil excellent        │
└──────────────┴──────────────────────────────────────────────┘
```

## Stack technique

| Composant | Rôle |
|---|---|
| [Streamlit](https://streamlit.io) | Interface web multipage |
| [scikit-learn](https://scikit-learn.org) | Modèle de classification (Pipeline, OneHotEncoder, RobustScaler) |
| [Plotly](https://plotly.com/python/) | Jauge de score, graphiques et carte interactive |
| [pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org) | Manipulation et génération des données |
| [joblib](https://joblib.readthedocs.io) | Sérialisation du modèle |

## Installation

```bash
git clone https://github.com/Kermah01/Credit-scoring.git
cd Credit-scoring

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. (Optionnel) Régénérer les données synthétiques — graine fixe, résultat identique
python generate_synthetic_data.py

# 2. (Optionnel) Ré-entraîner le modèle de scoring
python train_model.py

# 3. Lancer l'application
streamlit run app.py
```

L'application est alors disponible sur <http://localhost:8501>.

## Structure du projet

```
├── app.py                       # Page principale : scoring d'un dossier
├── pages/
│   └── 1_Tableau_de_bord.py     # Tableau de bord du portefeuille
├── features.py                  # Préparation des variables (feature engineering)
├── generate_synthetic_data.py   # Génération du jeu de données fictif (seed fixe)
├── train_model.py               # Entraînement et sélection du modèle
├── data/
│   ├── donnees_synthetiques.xlsx
│   └── agences.xlsx
├── model/
│   └── model.pkl                # Pipeline scikit-learn sérialisé
└── .streamlit/config.toml       # Thème de l'application
```

## Modèle

Le script `train_model.py` compare une **régression logistique** et une
**forêt aléatoire** par validation croisée (AUC ROC) et sauvegarde le meilleur
pipeline. Les variables utilisées reprennent la logique métier d'origine :
montant, durée et taux (mise à l'échelle robuste), profil de l'emprunteur et
caractéristiques du prêt (one-hot), tranche d'âge, garanties (indicateurs
binaires) et mois d'octroi.

## Licence

Projet de démonstration à but pédagogique.
