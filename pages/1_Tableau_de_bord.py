"""Tableau de bord interactif du portefeuille de prêts synthétique."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "donnees_synthetiques.xlsx"
AGENCES_PATH = ROOT / "data" / "agences.xlsx"

# Palette sobre validée (identité : ordre fixe, jamais recyclé)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PRIMARY = "#2a78d6"
CRITICAL = "#d03b3b"
SEQ_RISK = ["#fbe3e3", "#f3b0b0", "#e87f7e", "#d03b3b", "#9c2626"]
STATUT_COLORS = {"Sain": PRIMARY, "En défaut": CRITICAL}

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


def style_fig(fig, height: int = 420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, sans-serif", "color": "#52514e"},
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zeroline=False)
    return fig


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filtres simples dans la barre latérale."""
    st.sidebar.title("🎛️ Filtres")
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
    return out


def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    montant_moyen = f"{df['MONTANT SOLLICITE'].mean():,.0f}".replace(",", " ")
    c1.metric("Prêts", f"{len(df):,}".replace(",", " "))
    c2.metric("Montant moyen", f"{montant_moyen} FCFA")
    c3.metric("Durée moyenne", f"{df['DUREE DE REMBOURSEMENT'].mean():.0f} mois")
    taux_moyen = df["TAUX D'INTERET"].mean()
    c4.metric("Taux d'intérêt moyen", f"{taux_moyen:.2f} %")
    c5.metric("Taux de défaut", f"{df['STATUT'].mean():.1%}".replace(".", ","))


def tab_overview(df: pd.DataFrame) -> None:
    kpi_row(df)
    st.divider()

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
        map_style="carto-positron",
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
    st.title("📊 Tableau de bord du portefeuille")
    st.warning(
        "**Démonstration — données 100 % fictives**, générées synthétiquement "
        "à des fins de démonstration. Aucun client ni établissement réel.",
        icon="⚠️",
    )

    df, agences = load_data()
    df = apply_filters(df)

    if df.empty:
        st.info("Aucun prêt ne correspond aux filtres sélectionnés.")
        return

    t1, t2, t3, t4 = st.tabs(
        ["Vue d'ensemble", "Analyses", "Carte des agences", "Données"]
    )
    with t1:
        tab_overview(df)
    with t2:
        tab_analyses(df)
    with t3:
        tab_map(df, agences)
    with t4:
        tab_data(df)


if __name__ == "__main__":
    main()
