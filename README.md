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

- **Scoring interactif** (`app.py`) : saisie du dossier client dans la barre
  latérale (durée, montant, taux, sexe, situation matrimoniale, activité, type
  de prêt, agence, âge, garanties, mois d'octroi), puis bouton **Prédire** :
  probabilité de défaut affichée avec une jauge Plotly et un verdict
  (EXCELLENT / BON / MOYEN / ÉLEVÉ).
- **Tableau de bord interactif** (`pages/Dashboard.py`) : visualisation et
  filtrage de la base, camembert et histogramme, analyses croisées entre
  variables numériques et catégorielles, graphique chronologique et carte du
  taux de défaut par agence.

## Design

Le design est celui de la version d'origine de l'application : thème sombre,
bandeau-titre noir bordé de rouge, barre latérale noire bordée de rouge, photo
d'un immeuble de bureaux en fond de la page de scoring et fond néon sur le
tableau de bord (images chargées depuis leurs URL d'origine). Seuls ajouts :
une mention discrète « Données fictives — démonstration » sous le titre et
quelques règles CSS `@media` pour un affichage correct sur tablette et mobile
(fond en `background-size: cover`, sans `background-attachment: fixed`).

## Stack technique

| Composant | Rôle |
|---|---|
| [Streamlit](https://streamlit.io) | Interface web multipage |
| [scikit-learn](https://scikit-learn.org) | Modèle de classification (Pipeline, OneHotEncoder, RobustScaler) |
| [Plotly](https://plotly.com/python/) | Jauge de score, graphiques et carte interactive (fond Carto, sans jeton) |
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
│   └── Dashboard.py             # Tableau de bord interactif
├── features.py                  # Préparation des variables (feature engineering)
├── generate_synthetic_data.py   # Génération du jeu de données fictif (seed fixe)
├── train_model.py               # Entraînement et sélection du modèle
├── data/
│   ├── donnees_synthetiques.xlsx
│   └── agences.xlsx
├── model/
│   └── model.pkl                # Pipeline scikit-learn sérialisé
├── requirements.txt             # Versions épinglées (testées en Python 3.12 et 3.13)
└── .streamlit/config.toml       # Thème sombre
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

## Déploiement (Streamlit Community Cloud)

1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io) et connectez-vous avec votre compte GitHub.
2. Cliquez sur **New app**, puis choisissez ce dépôt, la branche à déployer et le fichier principal `app.py`.
3. Cliquez sur **Deploy** : l'application est construite puis mise en ligne sur une URL du type `https://<nom-de-l-appli>.streamlit.app`.

> **Version Python** : dans **Advanced settings** (avant le déploiement), choisissez
> Python 3.12 ou 3.13 : les versions épinglées dans `requirements.txt` ont été testées
> avec ces deux versions.
> Le modèle `model/model.pkl` étant sérialisé avec scikit-learn, gardez une version
> de Python et de scikit-learn cohérente avec celle de l'entraînement (relancez
> `train_model.py` en cas d'incompatibilité au chargement).

### Éviter l'hibernation

Streamlit Community Cloud met l'application en veille après environ 12 heures sans
trafic (un visiteur tombe alors sur un écran « l'appli se réveille » pendant
plusieurs dizaines de secondes). Pour l'éviter, ce dépôt contient le workflow
GitHub Actions [`.github/workflows/keep-alive.yml`](.github/workflows/keep-alive.yml)
qui envoie un ping HTTP à l'application toutes les 4 heures (cron `17 */4 * * *`).

Après le déploiement, renseignez l'URL de l'appli dans une variable de dépôt :

1. Sur GitHub : **Settings → Secrets and variables → Actions → Variables → New repository variable**.
2. Name : `APP_URL` — Value : l'URL publique de l'appli (ex. `https://<nom-de-l-appli>.streamlit.app`).

Tant que `APP_URL` n'est pas définie, le workflow se termine sans rien faire (et sans
échouer). À noter : GitHub désactive les workflows planifiés après 60 jours sans activité
sur le dépôt ; il suffit alors de le relancer une fois manuellement via l'onglet
**Actions → Keep-alive Streamlit → Run workflow**.

Alternative sans GitHub Actions : créer un moniteur HTTP(S) gratuit sur
[UptimeRobot](https://uptimerobot.com) qui interroge l'URL de l'appli toutes les 5 minutes.
