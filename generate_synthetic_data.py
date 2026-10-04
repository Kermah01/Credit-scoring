"""Génération d'un jeu de données 100 % fictif pour la démo de credit scoring.

Ce script crée, de façon entièrement reproductible (graine fixe), un portefeuille
de prêts bancaires synthétique : aucun client réel, aucune banque réelle, aucune
donnée personnelle. Les corrélations (taux d'intérêt, montant, durée, garanties)
sont injectées volontairement afin que le modèle de scoring ait quelque chose à
apprendre.

Sorties :
    - data/donnees_synthetiques.xlsx : le portefeuille de prêts fictif
    - data/agences.xlsx              : coordonnées (fictives) des agences

Usage :
    python generate_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_LOANS = 1200

DATA_DIR = Path(__file__).parent / "data"

SEXES = ["homme", "femme"]
SITUATIONS = ["célibataire", "marié(e)", "divorcé(e)", "veuf(ve)"]

SECTEURS = [
    "Administratif", "Agroalimentaire", "Assurance", "Banque", "Commerce",
    "Éducation", "Énergie", "Finance", "Informatique", "Juridique",
    "Marketing", "Production industrielle", "Professions libérales",
    "Ressources humaines", "Santé", "Transport",
]

PROFESSIONS = [
    "Ingénieur(e)", "Enseignant(e)", "Médecin", "Comptable", "Commerçant(e)",
    "Cadre administratif", "Technicien(ne)", "Consultant(e)", "Juriste",
    "Pharmacien(ne)", "Analyste financier", "Chef de projet",
    "Responsable commercial", "Infirmier(ère)", "Architecte",
]

TYPES_PRET = [
    "Prêt consommation", "Prêt immobilier", "Prêt scolarité", "Prêt véhicule",
    "Prêt équipement", "Prêt trésorerie", "Prêt travaux", "Prêt restructuration",
]

AGENCES = [
    "Agence Centre-Ville", "Agence du Marché", "Agence Université",
    "Agence Aéroport", "Agence de la Gare", "Agence Les Jardins",
    "Agence Bord de Mer", "Agence Zone Industrielle", "Agence Quartier Nord",
    "Agence des Collines",
]

CONSEILLERS = [f"Conseiller {i:02d}" for i in range(1, 13)]

GARANTIES = [
    "Domiciliation des revenus", "Cash collateral", "Dépôt à terme nanti",
    "Garantie hypothécaire", "Épargne bloquée", "Assurance décès/vie",
    "Billet à ordre", "Sans garantie",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_portfolio(n: int = N_LOANS, seed: int = SEED) -> pd.DataFrame:
    """Génère le portefeuille de prêts synthétique."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, size=n)
    sexe = rng.choice(SEXES, size=n, p=[0.62, 0.38])
    situation = rng.choice(SITUATIONS, size=n, p=[0.34, 0.52, 0.10, 0.04])
    secteur = rng.choice(SECTEURS, size=n)
    profession = rng.choice(PROFESSIONS, size=n)
    type_pret = rng.choice(
        TYPES_PRET, size=n,
        p=[0.30, 0.14, 0.10, 0.12, 0.08, 0.10, 0.10, 0.06],
    )
    agence = rng.choice(AGENCES, size=n)
    conseiller = rng.choice(CONSEILLERS, size=n)

    # Durée (mois) et montant (FCFA) dépendent du type de prêt.
    base_duree = {
        "Prêt consommation": 36, "Prêt immobilier": 120, "Prêt scolarité": 12,
        "Prêt véhicule": 48, "Prêt équipement": 36, "Prêt trésorerie": 12,
        "Prêt travaux": 60, "Prêt restructuration": 72,
    }
    base_montant = {
        "Prêt consommation": 3_000_000, "Prêt immobilier": 25_000_000,
        "Prêt scolarité": 1_000_000, "Prêt véhicule": 8_000_000,
        "Prêt équipement": 5_000_000, "Prêt trésorerie": 2_000_000,
        "Prêt travaux": 10_000_000, "Prêt restructuration": 7_000_000,
    }
    duree = np.array([
        int(np.clip(rng.normal(base_duree[t], base_duree[t] * 0.35), 6, 240))
        for t in type_pret
    ])
    montant = np.array([
        int(np.clip(rng.lognormal(np.log(base_montant[t]), 0.45), 200_000, 80_000_000))
        for t in type_pret
    ])
    montant = (montant // 10_000) * 10_000  # arrondi au 10 000 FCFA

    taux = np.round(np.clip(rng.normal(8.5, 2.0, size=n), 3.5, 14.0), 2)

    # Garanties : 1 à 3 garanties par prêt (ou "Sans garantie").
    garanties_list = []
    for _ in range(n):
        if rng.random() < 0.08:
            garanties_list.append(["Sans garantie"])
        else:
            k = rng.integers(1, 4)
            garanties_list.append(
                list(rng.choice(GARANTIES[:-1], size=k, replace=False))
            )
    garanties_str = ["; ".join(g) for g in garanties_list]

    # Date d'octroi entre 2015 et 2024.
    start = pd.Timestamp("2015-01-01")
    days = rng.integers(0, 3650, size=n)
    date_octroi = start + pd.to_timedelta(days, unit="D")

    # --- Variable cible : défaut de paiement (modèle logistique latent) ---
    z = -1.3
    z = z + 0.45 * (taux - 8.5)                       # taux élevé -> plus risqué
    z = z + 0.80 * (np.log(montant) - np.log(5e6))    # gros montants -> plus risqués
    z = z + 0.020 * (duree - 48)                      # longues durées -> plus risquées
    z = z - 0.030 * (age - 40)                        # clients plus âgés -> moins risqués

    protect = {"Garantie hypothécaire": 1.2, "Cash collateral": 1.0,
               "Dépôt à terme nanti": 0.9, "Épargne bloquée": 0.6,
               "Assurance décès/vie": 0.5, "Domiciliation des revenus": 0.5,
               "Billet à ordre": 0.2}
    z = z - np.array([sum(protect.get(g, 0.0) for g in gs) for gs in garanties_list])
    z = z + np.where(np.isin(type_pret, ["Prêt restructuration", "Prêt trésorerie"]), 1.3, 0.0)
    z = z + np.where(situation == "marié(e)", -0.4, 0.0)
    z = z + rng.normal(0, 0.35, size=n)               # bruit individuel

    statut = (rng.random(n) < _sigmoid(z)).astype(int)

    df = pd.DataFrame({
        "CLIENT": [f"CLT-{i:05d}" for i in range(1, n + 1)],
        "SEXE": sexe,
        "AGE": age,
        "SITUATION MATRIMONIALE": situation,
        "SECTEUR D'ACTIVITE": secteur,
        "PROFESSION": profession,
        "AGENCE": agence,
        "CONSEILLER": conseiller,
        "TYPE DE PRÊT": type_pret,
        "DUREE DE REMBOURSEMENT": duree,
        "MONTANT SOLLICITE": montant,
        "TAUX D'INTERET": taux,
        "GARANTIES": garanties_str,
        "DATE D'OCTROI": date_octroi,
        "STATUT": statut,
    })
    df["STATUT DU PRÊT"] = np.where(df["STATUT"] == 1, "En défaut", "Sain")
    return df


def generate_agencies(seed: int = SEED) -> pd.DataFrame:
    """Coordonnées fictives des agences, dispersées autour d'une ville imaginaire."""
    rng = np.random.default_rng(seed + 1)
    center_lat, center_lon = 5.32, -4.02  # ville côtière fictive
    return pd.DataFrame({
        "AGENCE": AGENCES,
        "latitude": np.round(center_lat + rng.uniform(-0.06, 0.06, len(AGENCES)), 5),
        "longitude": np.round(center_lon + rng.uniform(-0.08, 0.08, len(AGENCES)), 5),
    })


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_portfolio()
    agences = generate_agencies()

    df.to_excel(DATA_DIR / "donnees_synthetiques.xlsx", index=False)
    agences.to_excel(DATA_DIR / "agences.xlsx", index=False)

    taux_defaut = df["STATUT"].mean()
    print(f"{len(df)} prêts synthétiques générés (taux de défaut : {taux_defaut:.1%})")
    print(f"Fichiers écrits dans {DATA_DIR}/")


if __name__ == "__main__":
    main()
