"""Tableau de bord interactif du portefeuille de prêts synthétique."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from theme import (  # noqa: E402 — dépend du sys.path ci-dessus
    CATEGORICAL,
    PRIMARY,
    SEQ_RISK,
    STATUT_COLORS,
    demo_note,
    footer,
    hero,
    inject_css,
    kpi_band,
    sidebar_brand,
    sidebar_page_link,
    style_fig,
)

DATA_PATH = ROOT / "data" / "donnees_synthetiques.xlsx"
AGENCES_PATH = ROOT / "data" / "agences.xlsx"

MOIS_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet",
           "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

CAT_VARS = ["SEXE", "SITUATION MATRIMONIALE", "SECTEUR D'ACTIVITE", "PROFESSION",
            "AGENCE", "CONSEILLER", "TYPE DE PRÊT", "STATUT DU PRÊT",
            "MOIS D'OCTROI", "ANNEE D'OCTROI"]
NUM_VARS = ["AGE", "DUREE DE REMBOURSEMENT", "MONTANT SOLLICITE", "TAUX D'INTERET"]

st.set_page_config(
    page_title="Tableau de bord — Credit Scoring Démo",
    page_icon="📊",
    layout="wide",
)
inject_css()


@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(DATA_PATH)
    df["DATE D'OCTROI"] = pd.to_datetime(df["DATE D'OCTROI"])
    df["ANNEE D'OCTROI"] = df["DATE D'OCTROI"].dt.year
    df["MOIS D'OCTROI"] = pd.Categorical(
        df["DATE D'OCTROI"].dt.month.map(dict(enumerate(MOIS_FR, start=1))),
        categories=MOIS_FR, ordered=True,
    )
    agences = pd.read_excel(AGENCES_PATH)
    return df, agences


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filtres simples dans la barre latérale."""
    sidebar_brand("Tableau de bord")
    st.sidebar.markdown("#### Filtres")
    agences_sel = st.sidebar.multiselect("Agence", sorted(df["AGENCE"].unique()))
    types_sel = st.sidebar.multiselect("Type de prêt", sorted(df["TYPE DE PRÊT"].unique()))
    annees = sorted(df["ANNEE D'OCTROI"].unique())
    annee_min, annee_max = st.sidebar.select_slider(
        "Période d'octroi", options=annees, value=(annees[0], annees[-1]),
    )
    statut_sel = st.sidebar.multiselect("Statut du prêt", ["Sain", "En défaut"])

    out = df[(df["ANNEE D'OCTROI"] >= annee_min) & (df["ANNEE D'OCTROI"] <= annee_max)]
    if agences_sel:
        out = out[out["AGENCE"].isin(agences_sel)]
    if types_sel:
        out = out[out["TYPE DE PRÊT"].isin(types_sel)]
    if statut_sel:
        out = out[out["STATUT DU PRÊT"].isin(statut_sel)]

    sidebar_page_link("app.py", "Scorer un dossier", "🎯")
    st.sidebar.caption(
        "Portefeuille 100 % synthétique — aucun client ni établissement réel."
    )
    return out


def _fr(x: float, fmt: str) -> str:
    return format(x, fmt).replace(",", " ").replace(".", ",")


def kpi_row(df: pd.DataFrame, full: pd.DataFrame) -> None:
    """Bandeau KPI premium : valeur du périmètre filtré + delta vs portefeuille."""
    filtered = len(df) != len(full)

    def delta(cur: float, ref: float, fmt: str, unit: str, inverse: bool = False):
        if not filtered:
            return None, None
        d = cur - ref
        if abs(d) < 1e-9:
            return "±0", None
        sign = "+" if d > 0 else "−"
        good = (d < 0) if inverse else (d > 0)
        return f"{sign}{_fr(abs(d), fmt)}{unit}", good

    taux_moyen, taux_ref = df["TAUX D'INTERET"].mean(), full["TAUX D'INTERET"].mean()
    defaut, defaut_ref = df["STATUT"].mean(), full["STATUT"].mean()
    montant, montant_ref = df["MONTANT SOLLICITE"].mean(), full["MONTANT SOLLICITE"].mean()
    duree, duree_ref = df["DUREE DE REMBOURSEMENT"].mean(), full["DUREE DE REMBOURSEMENT"].mean()

    d_mt, g_mt = delta(montant / 1e6, montant_ref / 1e6, ".1f", " M")
    d_du, g_du = delta(duree, duree_ref, ".0f", " mois")
    d_tx, g_tx = delta(taux_moyen, taux_ref, ".2f", " pt", inverse=True)
    d_df, g_df = delta(defaut * 100, defaut_ref * 100, ".1f", " pt", inverse=True)

    kpi_band([
        {"icon": "folder", "label": "Prêts", "value": f"{len(df):,}".replace(",", " "),
         "sub": (f"{len(df) / len(full):.0%} du portefeuille" if filtered
                 else "portefeuille complet"),
         "delta": None},
        {"icon": "coins", "label": "Montant moyen",
         "value": _fr(montant / 1e6, ",.2f"), "unit": "M FCFA",
         "delta": d_mt, "good": g_mt, "sub": "vs portefeuille" if d_mt else None},
        {"icon": "clock", "label": "Durée moyenne",
         "value": f"{duree:.0f}", "unit": "mois",
         "delta": d_du, "good": g_du, "sub": "vs portefeuille" if d_du else None},
        {"icon": "percent", "label": "Taux d'intérêt moyen",
         "value": _fr(taux_moyen, ".2f"), "unit": "%",
         "delta": d_tx, "good": g_tx, "sub": "vs portefeuille" if d_tx else None},
        {"icon": "pulse", "label": "Taux de défaut",
         "value": _fr(defaut * 100, ".1f"), "unit": "%",
         "delta": d_df, "good": g_df, "sub": "vs portefeuille" if d_df else None},
    ])


def tab_overview(df: pd.DataFrame, full: pd.DataFrame) -> None:
    kpi_row(df, full)
    st.markdown("")

    chrono = (
        df.groupby(["ANNEE D'OCTROI", "STATUT DU PRÊT"], observed=True)
        .size().reset_index(name="Nombre de prêts")
    )
    fig = px.line(
        chrono, x="ANNEE D'OCTROI", y="Nombre de prêts", color="STATUT DU PRÊT",
        markers=True, color_discrete_map=STATUT_COLORS,
        title="Production de crédits par année et statut",
    )
    fig.update_traces(line_width=2, marker_size=8)
    st.plotly_chart(style_fig(fig), config={"displayModeBar": False})

    defaut_type = (
        df.groupby("TYPE DE PRÊT", observed=True)["STATUT"].mean()
        .sort_values().reset_index(name="Taux de défaut")
    )
    fig2 = px.bar(
        defaut_type, x="Taux de défaut", y="TYPE DE PRÊT", orientation="h",
        title="Taux de défaut par type de prêt",
        color_discrete_sequence=[PRIMARY],
    )
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(style_fig(fig2, 380), config={"displayModeBar": False})


def tab_analyses(df: pd.DataFrame) -> None:
    col_uni, col_croise = st.columns(2, gap="large")

    with col_uni:
        st.subheader("Analyse univariée")
        var = st.selectbox("Variable", CAT_VARS, index=6)
        counts = df[var].value_counts().sort_values()
        fig = px.bar(
            x=counts.values, y=counts.index.astype(str), orientation="h",
            labels={"x": "Nombre de prêts", "y": var},
            title=f"Répartition — {var}",
            color_discrete_sequence=[PRIMARY],
        )
        st.plotly_chart(style_fig(fig, 460), config={"displayModeBar": False})

    with col_croise:
        st.subheader("Analyse croisée")
        var1 = st.selectbox("Variable 1 (axe)", CAT_VARS, index=4, key="v1")
        var2 = st.selectbox(
            "Variable 2 (couleur)",
            ["STATUT DU PRÊT", "SEXE", "SITUATION MATRIMONIALE", "TYPE DE PRÊT"],
            key="v2",
        )
        mode = st.radio("Mode", ["Empilé", "Groupé"], horizontal=True)
        color_map = STATUT_COLORS if var2 == "STATUT DU PRÊT" else None
        fig = px.histogram(
            df, x=var1, color=var2,
            barmode="relative" if mode == "Empilé" else "group",
            color_discrete_sequence=CATEGORICAL,
            color_discrete_map=color_map,
            title=f"{var1} × {var2}",
        )
        fig.update_layout(yaxis_title="Nombre de prêts", bargap=0.25)
        st.plotly_chart(style_fig(fig, 460), config={"displayModeBar": False})

    st.subheader("Variables numériques")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        var_num = st.selectbox("Distribution de", NUM_VARS)
        fig = px.histogram(
            df, x=var_num, color="STATUT DU PRÊT",
            color_discrete_map=STATUT_COLORS, barmode="overlay", opacity=0.75,
            title=f"Distribution — {var_num}",
        )
        fig.update_layout(yaxis_title="Nombre de prêts", bargap=0.05)
        st.plotly_chart(style_fig(fig, 400), config={"displayModeBar": False})
    with c2:
        vx = st.selectbox("Axe X", NUM_VARS, index=2)
        vy = st.selectbox("Axe Y", NUM_VARS, index=3)
        fig = px.scatter(
            df, x=vx, y=vy, color="STATUT DU PRÊT",
            color_discrete_map=STATUT_COLORS, opacity=0.65,
            title=f"{vx} vs {vy}",
        )
        fig.update_traces(marker_size=8)
        st.plotly_chart(style_fig(fig, 400), config={"displayModeBar": False})


def tab_map(df: pd.DataFrame, agences: pd.DataFrame) -> None:
    st.subheader("Risque de défaut par agence")
    st.caption("Localisation fictive des agences, données synthétiques.")

    par_agence = (
        df.groupby("AGENCE", observed=True)
        .agg(**{
            "Nombre de prêts": ("STATUT", "size"),
            "Taux de défaut": ("STATUT", "mean"),
        })
        .reset_index()
        .merge(agences, on="AGENCE")
    )
    par_agence["Taux de défaut (%)"] = (par_agence["Taux de défaut"] * 100).round(1)

    fig = px.scatter_map(
        par_agence, lat="latitude", lon="longitude",
        size="Nombre de prêts", color="Taux de défaut",
        color_continuous_scale=SEQ_RISK, size_max=42, zoom=10.3,
        hover_name="AGENCE",
        hover_data={"latitude": False, "longitude": False,
                    "Taux de défaut": False,
                    "Nombre de prêts": True, "Taux de défaut (%)": True},
        map_style="carto-darkmatter",
    )
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Taux de défaut", tickformat=".0%"),
    )
    st.plotly_chart(fig, config={"displayModeBar": False})

    st.dataframe(
        par_agence[["AGENCE", "Nombre de prêts", "Taux de défaut (%)"]]
        .sort_values("Taux de défaut (%)", ascending=False),
        hide_index=True, width="stretch",
    )


def tab_data(df: pd.DataFrame) -> None:
    st.subheader("Portefeuille de prêts (données synthétiques)")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Télécharger (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="donnees_synthetiques.csv",
        mime="text/csv",
    )


def main() -> None:
    df_full, agences = load_data()
    volume = df_full["MONTANT SOLLICITE"].sum() / 1e9

    hero(
        "Tableau de bord du portefeuille",
        "Explorez le portefeuille de prêts synthétique : production, risque, "
        "analyses croisées et cartographie des agences fictives.",
        badges=[
            ("folder", f"{len(df_full):,}".replace(",", " "), "prêts"),
            ("coins", f"{volume:.1f}".replace(".", ","), "Mds FCFA d'encours"),
            ("map", f"{df_full['AGENCE'].nunique()}", "agences"),
        ],
    )
    demo_note(
        "<b>Démonstration — données 100 % fictives</b>, générées "
        "synthétiquement à des fins de démonstration. Aucun client ni "
        "établissement réel."
    )

    df = apply_filters(df_full)

    if df.empty:
        st.info("Aucun prêt ne correspond aux filtres sélectionnés.")
        return

    t1, t2, t3, t4 = st.tabs(
        ["Vue d'ensemble", "Analyses", "Carte des agences", "Données"]
    )
    with t1:
        tab_overview(df, df_full)
    with t2:
        tab_analyses(df)
    with t3:
        tab_map(df, agences)
    with t4:
        tab_data(df)

    footer(
        "Projet de démonstration — scoring de crédit par apprentissage automatique.",
        "Scorez un dossier dans la page <b>Credit Scoring</b>.",
    )


if __name__ == "__main__":
    main()
