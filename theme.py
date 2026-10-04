"""Thème partagé « Nuit électrique » — design system de la démo Credit Scoring.

Centralise, pour les DEUX pages de l'application :

- les jetons de couleur (fonds, encres, accents électriques) ;
- les palettes de graphiques **validées** avec le validateur du skill dataviz
  (mode sombre, surface de référence ``#0e1433``) :

  * catégorielle 8 teintes (ordre fixe, jamais recyclé) — ALL CHECKS PASS ;
  * rampe séquentielle rouge du risque — ALL CHECKS PASS (ordinal, mono-teinte,
    extrémité claire >= 2:1 sur la surface) ;
  * statuts (good/warning/serious/critical) — palette réservée, >= 3:1 sur la
    surface sombre, toujours accompagnée d'une étiquette/position, jamais seule.

- le template Plotly sombre commun (enregistré par défaut à l'import) ;
- l'injection CSS : fond animé pleine page (gradient en keyframes + blobs
  lumineux en pur CSS, aucune image externe), glassmorphism, typographie
  Google Fonts, micro-interactions ;
- le « hero » d'entrée de page avec badge « données fictives ».
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Jetons de couleur — ambiance fintech bleu nuit / indigo électrique
# ---------------------------------------------------------------------------
BG_PAGE = "#060a1f"          # plan de page (config.toml backgroundColor)
SURFACE = "#0e1433"          # surface de graphique de référence (validateur)
SURFACE_2 = "#101735"        # fond secondaire (widgets, sidebar)

INK = "#eef1ff"              # encre primaire
INK_SECONDARY = "#a7b0d8"    # encre secondaire
MUTED = "#7681a8"            # axes, libellés discrets
GRID = "rgba(167, 176, 216, 0.14)"
AXIS = "rgba(167, 176, 216, 0.30)"

ACCENT = "#7c8cf8"           # indigo électrique (primaryColor)
ACCENT_CYAN = "#5fe0ff"
ACCENT_VIOLET = "#b79cff"

# Palette catégorielle sombre — ordre fixe, validée (adjacent, mode sombre)
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]
PRIMARY = CATEGORICAL[0]

# Statuts réservés (jauge de risque) — fixes, >= 3:1 sur la surface sombre
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_SERIOUS = "#ec835a"
COLOR_CRITICAL = "#d03b3b"

# Série « statut du prêt » : l'identité suit l'entité, jamais le rang
STATUT_COLORS = {"Sain": CATEGORICAL[0], "En défaut": CATEGORICAL[7]}
CRITICAL = CATEGORICAL[7]

# Rampe séquentielle du risque (carte) — rouge mono-teinte, sombre -> brillant
SEQ_RISK = ["#853340", "#a93c45", "#cd4a4e", "#ee6355", "#ff9479"]

FONT_BODY = "'Inter', 'Segoe UI', system-ui, sans-serif"
FONT_DISPLAY = "'Space Grotesk', 'Inter', system-ui, sans-serif"

# ---------------------------------------------------------------------------
# Template Plotly sombre commun — enregistré comme défaut à l'import
# ---------------------------------------------------------------------------
pio.templates["nuit_electrique"] = go.layout.Template(
    layout=go.Layout(
        colorway=CATEGORICAL,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color=INK_SECONDARY, size=13),
        title=dict(font=dict(family=FONT_DISPLAY, color=INK, size=17)),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                   tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False,
                   tickfont=dict(color=MUTED), title_font=dict(color=INK_SECONDARY)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY)),
        hoverlabel=dict(bgcolor="#141b42", bordercolor="rgba(124,140,248,0.45)",
                        font=dict(family=FONT_BODY, color=INK, size=13)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
)
pio.templates.default = "nuit_electrique"


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    """Finitions communes des graphiques (hauteur, légende horizontale)."""
    fig.update_layout(
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# CSS global — fond animé, glassmorphism, micro-interactions
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ----- Typographie ----- */
html, body, .stApp, [data-testid="stSidebar"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
h1, h2, h3, h4, [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif !important;
    letter-spacing: -0.01em;
}

/* ----- Fond immersif : gradient animé + blobs lumineux (pur CSS) ----- */
.stApp {
    background: linear-gradient(235deg,
        #0b1134 0%, #060a1f 30%, #150f3d 55%, #081228 80%, #0b1134 100%);
    background-size: 380% 380%;
    animation: aurora 42s ease-in-out infinite;
}
@keyframes aurora {
    0%   { background-position: 0% 40%; }
    50%  { background-position: 100% 60%; }
    100% { background-position: 0% 40%; }
}
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    inset: -22%;
    pointer-events: none;
    z-index: 0;
    will-change: transform;
}
.stApp::before {
    background:
        radial-gradient(42% 36% at 22% 26%, rgba(99, 91, 255, 0.34), transparent 70%),
        radial-gradient(34% 30% at 80% 18%, rgba(45, 192, 235, 0.16), transparent 70%),
        radial-gradient(46% 42% at 72% 80%, rgba(168, 85, 247, 0.20), transparent 72%);
    animation: blob-drift-a 28s ease-in-out infinite alternate;
}
.stApp::after {
    background:
        radial-gradient(30% 26% at 28% 78%, rgba(57, 135, 229, 0.22), transparent 70%),
        radial-gradient(24% 22% at 88% 52%, rgba(14, 165, 233, 0.13), transparent 70%);
    animation: blob-drift-b 36s ease-in-out infinite alternate;
}
@keyframes blob-drift-a {
    from { transform: translate3d(-3%, -2%, 0) scale(1) rotate(0deg); }
    to   { transform: translate3d(4%, 3%, 0) scale(1.14) rotate(7deg); }
}
@keyframes blob-drift-b {
    from { transform: translate3d(3%, 2%, 0) scale(1.08) rotate(5deg); }
    to   { transform: translate3d(-4%, -3%, 0) scale(0.98) rotate(-4deg); }
}
@media (prefers-reduced-motion: reduce) {
    .stApp, .stApp::before, .stApp::after { animation: none !important; }
}
/* Contenu au-dessus des couches de fond */
[data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stHeader"] {
    position: relative;
    z-index: 1;
}
[data-testid="stHeader"] { background: transparent; }

/* ----- Sidebar : verre sombre ----- */
[data-testid="stSidebar"] {
    background: rgba(7, 11, 31, 0.66);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid rgba(148, 163, 255, 0.12);
}
[data-testid="stSidebar"] [data-testid="stForm"] {
    background: rgba(148, 163, 255, 0.05);
    border: 1px solid rgba(148, 163, 255, 0.13);
    border-radius: 18px;
    padding: 1rem 1rem 1.2rem;
}

/* ----- Cartes de verre : métriques ----- */
[data-testid="stMetric"] {
    background: rgba(148, 163, 255, 0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 255, 0.14);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: 0 10px 34px rgba(3, 6, 20, 0.45);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(124, 140, 248, 0.45);
    box-shadow: 0 16px 44px rgba(63, 73, 180, 0.35);
}
[data-testid="stMetricLabel"] p { color: #a7b0d8 !important; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #eef1ff; }

/* ----- Cartes de verre : graphiques, expandeurs, alertes, tableaux ----- */
[data-testid="stPlotlyChart"] {
    background: rgba(13, 18, 48, 0.52);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(148, 163, 255, 0.12);
    border-radius: 20px;
    padding: 0.65rem;
    box-shadow: 0 12px 40px rgba(3, 6, 20, 0.45);
    transition: border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(124, 140, 248, 0.32);
    box-shadow: 0 16px 48px rgba(63, 73, 180, 0.28);
}
[data-testid="stExpander"] details {
    background: rgba(148, 163, 255, 0.05);
    border: 1px solid rgba(148, 163, 255, 0.13);
    border-radius: 16px;
}
[data-testid="stAlertContainer"], [data-testid="stAlert"] {
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 255, 0.13);
    border-radius: 16px;
    overflow: hidden;
}

/* ----- Boutons : dégradé électrique + élévation au survol ----- */
.stButton > button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button, [data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #4f46e5 0%, #6d5ef1 55%, #3b82f6 100%) !important;
    color: #f5f7ff !important;
    border: 1px solid rgba(165, 180, 252, 0.35) !important;
    border-radius: 14px !important;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 26px rgba(79, 70, 229, 0.38);
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover, [data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    filter: brightness(1.12);
    box-shadow: 0 14px 36px rgba(99, 102, 241, 0.5);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0);
}

/* ----- Onglets ----- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(148, 163, 255, 0.06);
    border: 1px solid rgba(148, 163, 255, 0.12);
    border-radius: 16px;
    padding: 0.3rem;
    width: fit-content;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    border-radius: 12px;
    padding: 0.35rem 1.05rem;
    color: #a7b0d8;
    transition: background .2s ease, color .2s ease;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    background: rgba(124, 140, 248, 0.14);
    color: #eef1ff;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, rgba(79, 70, 229, 0.55), rgba(59, 130, 246, 0.45));
    color: #ffffff;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* ----- Hero ----- */
.hero { padding: 0.6rem 0 0.9rem; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.32rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(250, 178, 25, 0.45);
    background: rgba(250, 178, 25, 0.10);
    color: #ffd98a;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-badge-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #fab219;
    box-shadow: 0 0 10px rgba(250, 178, 25, 0.9);
    animation: badge-pulse 2.4s ease-in-out infinite;
}
@keyframes badge-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.8); }
}
.hero-title {
    margin: 0.75rem 0 0.35rem;
    font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif;
    font-size: clamp(2.3rem, 4.6vw, 3.5rem);
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -0.02em;
    background: linear-gradient(92deg, #8ab6ff 0%, #b79cff 48%, #5fe0ff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
.hero-sub {
    margin: 0;
    max-width: 62ch;
    color: #a7b0d8;
    font-size: 1.02rem;
    line-height: 1.55;
}

/* ----- Divers ----- */
hr { border-color: rgba(148, 163, 255, 0.14); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 255, 0.22);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 255, 0.38); }
</style>
"""


def inject_css() -> None:
    """Injecte le thème global (fond animé, verre, typographie) dans la page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str,
         badge: str = "Démo — données 100 % fictives") -> None:
    """Hero d'entrée de page : badge, grand titre en dégradé, sous-titre."""
    st.markdown(
        f"""
        <div class="hero">
          <span class="hero-badge"><span class="hero-badge-dot"></span>{badge}</span>
          <h1 class="hero-title">{title}</h1>
          <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
