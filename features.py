"""Préparation des variables pour le modèle de credit scoring.

Logique métier conservée de la version d'origine :
    - variables numériques (durée, montant, taux) mises à l'échelle robuste ;
    - variables catégorielles encodées en one-hot ;
    - âge discrétisé en tranches ;
    - garanties encodées en indicateurs binaires ;
    - mois d'octroi du crédit conservé comme variable.

L'encodage (RobustScaler + OneHotEncoder) est porté par un Pipeline scikit-learn,
ce qui garantit la cohérence entre entraînement et prédiction.
"""

import pandas as pd

GARANTIES = [
    "Domiciliation des revenus", "Cash collateral", "Dépôt à terme nanti",
    "Garantie hypothécaire", "Épargne bloquée", "Assurance décès/vie",
    "Billet à ordre", "Sans garantie",
]

AGE_BINS = [18, 30, 40, 50, 60, 120]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]

NUMERIC_FEATURES = ["DUREE DE REMBOURSEMENT", "MONTANT SOLLICITE", "TAUX D'INTERET"]
CATEGORICAL_FEATURES = [
    "SEXE", "SITUATION MATRIMONIALE", "SECTEUR D'ACTIVITE",
    "TYPE DE PRÊT", "AGENCE", "TRANCHE D'AGE",
]
GARANTIE_FEATURES = [f"GAR_{g}" for g in GARANTIES]
MONTH_FEATURE = ["MOIS D'OCTROI"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + GARANTIE_FEATURES + MONTH_FEATURE


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice de variables attendue par le pipeline de scoring.

    Attend un DataFrame contenant au moins : les colonnes numériques et
    catégorielles brutes, ``AGE``, ``GARANTIES`` (chaîne "g1; g2; ...")
    et ``DATE D'OCTROI`` ou ``MOIS D'OCTROI``.
    """
    out = df.copy()

    out["TRANCHE D'AGE"] = pd.cut(
        out["AGE"].astype(int), bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)

    garanties = out["GARANTIES"].fillna("").astype(str)
    for g in GARANTIES:
        out[f"GAR_{g}"] = garanties.str.contains(g, regex=False).astype(int)

    if "MOIS D'OCTROI" not in out.columns:
        out["MOIS D'OCTROI"] = pd.to_datetime(out["DATE D'OCTROI"]).dt.month

    return out[ALL_FEATURES]
