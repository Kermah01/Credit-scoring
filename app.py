"""Credit Scoring App — Démo : page de scoring d'une demande de prêt.

Application de démonstration : toutes les données sont synthétiques
(générées par ``generate_synthetic_data.py``) et aucun établissement
bancaire réel n'est représenté.
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from features import GARANTIES, build_features
from generate_synthetic_data import (
    AGENCES,
    SECTEURS,
    SEXES,
    SITUATIONS,
    TYPES_PRET,
)
from theme import (
    AXIS,
    COLOR_CRITICAL,
    COLOR_GOOD,
    COLOR_SERIOUS,
    COLOR_WARNING,
    FONT_DISPLAY,
    INK,
    MUTED,
    hero,
    inject_css,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.pkl"

st.set_page_config(
    page_title="Credit Scoring App — Démo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model():
    return joblib.load(MODEL_PATH)


def disclaimer() -> None:
    st.warning(
        "**Démonstration — données 100 % fictives.** "
        "L'ensemble des données (clients, agences, prêts) est généré "
        "synthétiquement à des fins de démonstration ; aucune donnée réelle "
        "ni aucun établissement bancaire réel n'est représenté. "
        "Les scores produits n'ont aucune valeur de décision de crédit.",
        icon="⚠️",
    )


def gauge(prob: float) -> go.Figure:
    """Jauge de probabilité de défaut (0–100 %)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={
                "suffix": " %",
                "font": {"size": 46, "color": INK, "family": FONT_DISPLAY},
            },
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": AXIS,
                    "tickfont": {"color": MUTED},
                },
                "bar": {"color": INK, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": COLOR_GOOD},
                    {"range": [25, 50], "color": COLOR_WARNING},
                    {"range": [50, 70], "color": COLOR_SERIOUS},
                    {"range": [70, 100], "color": COLOR_CRITICAL},
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig


def risk_label(prob: float) -> tuple[str, str]:
    """Retourne (niveau de risque, type d'alerte Streamlit)."""
    if prob < 25:
        return "Risque faible — profil excellent", "success"
    if prob < 50:
        return "Risque modéré — profil bon", "success"
    if prob < 70:
        return "Risque significatif — vigilance recommandée", "warning"
    return "Risque élevé — crédit potentiellement risqué", "error"


def sidebar_inputs() -> dict:
    st.sidebar.title("📋 Dossier de demande")
    st.sidebar.caption("Renseignez le profil de l'emprunteur fictif.")

    with st.sidebar.form("dossier"):
        st.subheader("Prêt")
        montant = st.number_input(
            "Montant du prêt (FCFA)", min_value=100_000, max_value=100_000_000,
            value=5_000_000, step=100_000,
        )
        duree = st.slider("Durée de remboursement (mois)", 6, 240, 48)
        taux = st.number_input(
            "Taux d'intérêt (%)", min_value=0.0, max_value=25.0, value=8.5, step=0.25,
        )
        type_pret = st.selectbox("Type de prêt", TYPES_PRET)
        mois = st.slider("Mois d'octroi du crédit", 1, 12, 6)

        st.subheader("Emprunteur")
        age = st.number_input("Âge (années)", min_value=18, max_value=90, value=35)
        sexe = st.selectbox("Sexe", SEXES)
        situation = st.selectbox("Situation matrimoniale", SITUATIONS)
        secteur = st.selectbox("Secteur d'activité", SECTEURS)
        agence = st.selectbox("Agence", AGENCES)
        garanties = st.multiselect("Garanties proposées", GARANTIES[:-1])

        submitted = st.form_submit_button("Calculer le score", width="stretch", type="primary")

    return {
        "submitted": submitted,
        "row": pd.DataFrame([{
            "DUREE DE REMBOURSEMENT": duree,
            "MONTANT SOLLICITE": montant,
            "TAUX D'INTERET": taux,
            "SEXE": sexe,
            "SITUATION MATRIMONIALE": situation,
            "SECTEUR D'ACTIVITE": secteur,
            "TYPE DE PRÊT": type_pret,
            "AGENCE": agence,
            "AGE": age,
            "GARANTIES": "; ".join(garanties) if garanties else "Sans garantie",
            "MOIS D'OCTROI": mois,
        }]),
    }


def main() -> None:
    hero(
        "Credit Scoring",
        "Estimez en un clic la probabilité de défaut de paiement d'un emprunteur, "
        "grâce à un modèle de machine learning entraîné sur un portefeuille "
        "entièrement synthétique.",
    )
    disclaimer()

    model = load_model()
    inputs = sidebar_inputs()

    if inputs["submitted"]:
        X = build_features(inputs["row"])
        prob = float(model.predict_proba(X)[0, 1]) * 100

        label, level = risk_label(prob)

        col_gauge, col_detail = st.columns([3, 2], gap="large")
        with col_gauge:
            st.subheader("Probabilité de défaut de paiement")
            st.plotly_chart(gauge(prob), config={"displayModeBar": False})
        with col_detail:
            st.subheader("Synthèse")
            row = inputs["row"].iloc[0]
            montant_txt = f"{row['MONTANT SOLLICITE']:,.0f}".replace(",", " ")
            taux_val = row["TAUX D'INTERET"]
            m1, m2 = st.columns(2)
            m1.metric("Probabilité de défaut", f"{prob:.1f} %")
            m2.metric("Montant demandé", f"{montant_txt} FCFA")
            m3, m4 = st.columns(2)
            m3.metric("Durée", f"{row['DUREE DE REMBOURSEMENT']} mois")
            m4.metric("Taux d'intérêt", f"{taux_val} %")

            if level == "success":
                st.success(label, icon="✅")
            elif level == "warning":
                st.warning(label, icon="⚠️")
            else:
                st.error(label, icon="🚨")

        with st.expander("Voir les données transmises au modèle"):
            st.dataframe(inputs["row"], width="stretch", hide_index=True)
    else:
        st.info(
            "Renseignez le dossier dans la barre latérale puis cliquez sur "
            "**Calculer le score** pour obtenir la probabilité de défaut.",
            icon="👈",
        )
        st.plotly_chart(gauge(0), config={"displayModeBar": False})

    st.divider()
    st.caption(
        "Projet de démonstration — Scoring de crédit par apprentissage automatique. "
        "Explorez le portefeuille synthétique dans la page **Tableau de bord**."
    )


if __name__ == "__main__":
    main()
