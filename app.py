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
from sklearn.metrics import roc_auc_score

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
    demo_note,
    footer,
    hero,
    icon,
    inject_css,
    section_title,
    sidebar_brand,
    sidebar_page_link,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.pkl"
DATA_PATH = ROOT / "data" / "donnees_synthetiques.xlsx"

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


@st.cache_data(show_spinner=False)
def portfolio_stats() -> dict:
    """Statistiques du portefeuille synthétique pour les badges du hero."""
    df = pd.read_excel(DATA_PATH)
    proba = load_model().predict_proba(build_features(df))[:, 1]
    return {
        "n": len(df),
        "defaut": float(df["STATUT"].mean()),
        "auc": float(roc_auc_score(df["STATUT"], proba)),
    }


def gauge(prob: float) -> go.Figure:
    """Jauge de probabilité de défaut (0–100 %)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={
                "suffix": " %",
                "font": {"size": 44, "color": INK, "family": FONT_DISPLAY},
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
    fig.update_layout(height=290, margin=dict(l=30, r=30, t=40, b=6))
    return fig


def risk_verdict(prob: float) -> dict:
    """Niveau de risque, classe de style et recommandation métier."""
    if prob < 25:
        return {
            "label": "Risque faible", "css": "v-good", "icon": "shield",
            "reco": "Profil excellent : la probabilité de défaut estimée est "
                    "faible. Le dossier peut être instruit favorablement, aux "
                    "conditions standard de l'établissement.",
        }
    if prob < 50:
        return {
            "label": "Risque modéré", "css": "v-warn", "icon": "scale",
            "reco": "Bon profil : le risque reste maîtrisé. Un avis favorable "
                    "est envisageable, en confirmant la capacité de "
                    "remboursement et les garanties proposées.",
        }
    if prob < 70:
        return {
            "label": "Risque significatif", "css": "v-serious", "icon": "pulse",
            "reco": "Vigilance recommandée : renforcer les garanties, ajuster "
                    "le montant ou la durée, et soumettre le dossier à une "
                    "revue approfondie avant décision.",
        }
    return {
        "label": "Risque élevé", "css": "v-crit", "icon": "zap",
        "reco": "Crédit potentiellement risqué : la probabilité de défaut "
                "estimée est très élevée. Un refus ou une restructuration "
                "profonde du dossier est recommandé.",
    }


def scoring_form() -> dict:
    """Formulaire de scoring en carte de verre, structuré en trois sections."""
    with st.form("dossier"):
        section_title("user", "Emprunteur", "Profil du demandeur")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge (années)", min_value=18, max_value=90, value=35)
            situation = st.selectbox("Situation matrimoniale", SITUATIONS)
            agence = st.selectbox("Agence", AGENCES)
        with c2:
            sexe = st.selectbox("Sexe", SEXES)
            secteur = st.selectbox("Secteur d'activité", SECTEURS)

        section_title("card", "Crédit", "Caractéristiques du prêt sollicité")
        c3, c4 = st.columns(2)
        with c3:
            montant = st.number_input(
                "Montant du prêt (FCFA)", min_value=100_000, max_value=100_000_000,
                value=5_000_000, step=100_000,
            )
            taux = st.number_input(
                "Taux d'intérêt (%)", min_value=0.0, max_value=25.0, value=8.5, step=0.25,
            )
        with c4:
            type_pret = st.selectbox("Type de prêt", TYPES_PRET)
            mois = st.slider("Mois d'octroi du crédit", 1, 12, 6)
        duree = st.slider("Durée de remboursement (mois)", 6, 240, 48)

        section_title("shield", "Garanties", "Sûretés proposées à l'appui du dossier")
        garanties = st.multiselect(
            "Garanties proposées", GARANTIES[:-1],
            placeholder="Sélectionnez une ou plusieurs garanties…",
        )

        submitted = st.form_submit_button(
            "Calculer le score", width="stretch", type="primary"
        )

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


def verdict_card(prob: float, row: pd.Series) -> None:
    """Carte verdict premium : badge, probabilité énorme, recommandation."""
    v = risk_verdict(prob)
    montant_txt = f"{row['MONTANT SOLLICITE']:,.0f}".replace(",", " ")
    st.markdown(
        f"""
        <div class="verdict">
          <span class="verdict-badge {v['css']}">{icon(v['icon'], 14)}{v['label']}</span>
          <div class="verdict-prob">{f"{prob:.1f}".replace(".", ",")}<small> %</small></div>
          <div class="verdict-cap">Probabilité de défaut estimée</div>
          <p class="verdict-msg">{v['reco']}</p>
          <div class="verdict-grid">
            <div><div class="verdict-cell-lab">Montant</div>
                 <div class="verdict-cell-val">{montant_txt} FCFA</div></div>
            <div><div class="verdict-cell-lab">Durée</div>
                 <div class="verdict-cell-val">{row['DUREE DE REMBOURSEMENT']} mois</div></div>
            <div><div class="verdict-cell-lab">Taux</div>
                 <div class="verdict-cell-val">{str(row["TAUX D'INTERET"]).replace(".", ",")} %</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_content(stats: dict) -> None:
    sidebar_brand()
    st.sidebar.markdown(
        f"""
        <div class="side-card">
          <div class="side-card-title">{icon("sparkles", 15)} Modèle de scoring</div>
          <div class="side-row"><span>Algorithme</span><b>Forêt aléatoire</b></div>
          <div class="side-row"><span>AUC ROC</span><b>{stats['auc']:.2f}</b></div>
          <div class="side-row"><span>Dossiers d'entraînement</span>
            <b>{f"{stats['n']:,}".replace(",", " ")}</b></div>
          <div class="side-row"><span>Taux de défaut observé</span>
            <b>{f"{stats['defaut']:.1%}".replace(".", ",")}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sidebar_page_link(
        "pages/1_Tableau_de_bord.py", "Explorer le tableau de bord", "📊"
    )
    st.sidebar.caption(
        "Portefeuille 100 % synthétique — aucun client ni établissement réel."
    )


def main() -> None:
    stats = portfolio_stats()

    hero(
        "Scoring de crédit, en un regard",
        "Estimez en un clic la probabilité de défaut de paiement d'un "
        "emprunteur, grâce à un modèle de machine learning entraîné sur un "
        "portefeuille entièrement synthétique.",
        badges=[
            ("target", f"{stats['auc']:.2f}", "AUC ROC"),
            ("folder", f"{stats['n']:,}".replace(",", " "), "dossiers scorés"),
            ("pulse", f"{stats['defaut']:.1%}".replace(".", ","), "taux de défaut"),
        ],
    )
    demo_note(
        "<b>Démonstration — données 100 % fictives.</b> Clients, agences et "
        "prêts sont générés synthétiquement ; aucune donnée réelle ni aucun "
        "établissement bancaire réel n'est représenté. Les scores produits "
        "n'ont aucune valeur de décision de crédit."
    )

    model = load_model()
    sidebar_content(stats)

    col_form, col_res = st.columns([11, 9], gap="large")

    with col_form:
        inputs = scoring_form()

    with col_res:
        if inputs["submitted"]:
            X = build_features(inputs["row"])
            prob = float(model.predict_proba(X)[0, 1]) * 100
            verdict_card(prob, inputs["row"].iloc[0])
            st.plotly_chart(gauge(prob), config={"displayModeBar": False})
        else:
            st.markdown(
                f"""
                <div class="empty-state">
                  <span class="sec-ic">{icon("gauge", 24)}</span>
                  <div class="empty-state-t">Prêt à scorer un dossier</div>
                  <div class="empty-state-s">
                    Renseignez le profil de l'emprunteur, les caractéristiques
                    du crédit et les garanties, puis cliquez sur
                    <b>Calculer le score</b> : la probabilité de défaut et la
                    recommandation s'afficheront ici.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(gauge(0), config={"displayModeBar": False})

    if inputs["submitted"]:
        with st.expander("Voir les données transmises au modèle"):
            st.dataframe(inputs["row"], width="stretch", hide_index=True)

    footer(
        "Projet de démonstration — scoring de crédit par apprentissage automatique.",
        "Explorez le portefeuille dans la page <b>Tableau de bord</b>.",
    )


if __name__ == "__main__":
    main()
