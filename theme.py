"""Thème partagé « Skyline » — design system de la démo Credit Scoring.

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
- le fond photographique pleine page (``assets/hero_bg.webp`` encodé en
  base64 + scrim dégradé : la skyline reste visible en haut, le bas devient
  quasi opaque pour la lisibilité) ;
- l'injection CSS : glassmorphism, typographie Google Fonts, micro-interactions ;
- les composants HTML partagés : hero à badges de verre, titres de section
  iconés, bandeau KPI premium, carte verdict, note « données fictives »,
  bloc de marque de la sidebar et footer.
"""

import base64
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).parent
BG_IMAGE = ROOT / "assets" / "hero_bg.webp"

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

# Encres de statut pour petit texte sur fond sombre (UI, jamais des marques)
TEXT_GOOD = "#5fd79a"
TEXT_BAD = "#ff8a7a"

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
        margin=dict(l=10, r=10, t=86, b=10),
        title=dict(y=1.0, yanchor="top", pad=dict(t=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Icônes SVG inline (trait fin, couleur héritée) — identité « expert »
# ---------------------------------------------------------------------------
_SVG = ('<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round">{body}</svg>')

_ICONS = {
    "user": '<circle cx="12" cy="8" r="3.6"/><path d="M5 20c1.4-3.6 3.9-5.2 7-5.2s5.6 1.6 7 5.2"/>',
    "card": '<rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3 10h18M7 15h4"/>',
    "shield": '<path d="M12 3l7 2.8v5.4c0 4.4-2.9 7.4-7 9-4.1-1.6-7-4.6-7-9V5.8z"/><path d="M9.2 12l2 2 3.6-3.8"/>',
    "bank": '<path d="M3 9.5L12 3l9 6.5"/><path d="M5 10v8M9.6 10v8M14.4 10v8M19 10v8M3 20.5h18"/>',
    "chart": '<path d="M4 20h16"/><path d="M6.5 20v-6M11.5 20V8M16.5 20v-9"/><path d="M5 6.5l5-2.5 4.5 3L20 4"/>',
    "pulse": '<path d="M3 12h4l2.5-6 4 12 2.5-6h5"/>',
    "coins": '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v5c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/><path d="M4.5 11v5c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-5"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2.2"/>',
    "percent": '<path d="M18.5 5.5l-13 13"/><circle cx="7.5" cy="7.5" r="2.4"/><circle cx="16.5" cy="16.5" r="2.4"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.6"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4.2l2 2.4H19a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "zap": '<path d="M13 2.5L4.5 13.5H11l-1.2 8L18.5 10.5H12z"/>',
    "gauge": '<path d="M4.5 19a8.8 8.8 0 1 1 15 0"/><path d="M12 15l3.6-4.4"/><circle cx="12" cy="15" r="1.1" fill="currentColor"/>',
    "map": '<path d="M12 21.5s-6.6-5.4-6.6-10.4a6.6 6.6 0 1 1 13.2 0c0 5-6.6 10.4-6.6 10.4z"/><circle cx="12" cy="10.8" r="2.3"/>',
    "sparkles": '<path d="M12 4l1.7 4.3L18 10l-4.3 1.7L12 16l-1.7-4.3L6 10l4.3-1.7z"/><path d="M19 15.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/>',
    "scale": '<path d="M12 4v16M8 20h8"/><path d="M12 6l-6 2 6-2 6 2"/><path d="M6 8l-2.6 6a3 3 0 0 0 5.2 0zM18 8l-2.6 6a3 3 0 0 0 5.2 0z"/>',
}


def icon(name: str, size: int = 16) -> str:
    """Retourne le SVG inline d'une icône du thème."""
    return _SVG.format(s=size, body=_ICONS[name])


# ---------------------------------------------------------------------------
# Fond photographique — encodé une seule fois par session serveur
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _bg_data_uri() -> str:
    """Encode assets/hero_bg.webp (skyline nocturne) en data-URI base64."""
    return base64.b64encode(BG_IMAGE.read_bytes()).decode()


# ---------------------------------------------------------------------------
# CSS global — fond image + scrim, verre, typographie, composants
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
h2, h3 { color: #eef1ff !important; }
[data-testid="stMain"] .block-container {
    max-width: 1280px;
    padding-top: 2.1rem;
    padding-bottom: 3.5rem;
}

/* ----- Fond immersif : skyline photoréaliste + scrim dégradé -----
   Le haut (50 %) laisse respirer la skyline, le bas devient quasi opaque
   pour garantir la lisibilité du contenu long. L'image est fixe (parallaxe
   douce au défilement), le scrim suit le viewport. */
.stApp {
    background:
        linear-gradient(180deg,
            rgba(4, 8, 18, 0.28) 0%,
            rgba(4, 8, 18, 0.50) 48%,
            rgba(5, 9, 20, 0.84) 78%,
            rgba(5, 9, 20, 0.95) 100%),
        url(data:image/webp;base64,__BG64__) center bottom / auto 150% fixed no-repeat;
    background-color: #060a1f;
}
@media (min-aspect-ratio: 12/5) {
    .stApp { background-size: auto, cover; }
}
[data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stHeader"] {
    position: relative;
    z-index: 1;
}
[data-testid="stHeader"] { background: transparent; }

/* Navigation native masquée : la sidebar porte ses propres liens stylés */
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebarUserContent"] { padding-top: 1.4rem; }

/* ----- Sidebar : verre sombre ----- */
[data-testid="stSidebar"] {
    background: rgba(6, 10, 28, 0.72);
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
.side-brand {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.35rem 0 0.9rem;
    border-bottom: 1px solid rgba(148, 163, 255, 0.12);
    margin-bottom: 0.4rem;
}
.side-brand-logo {
    width: 42px; height: 42px; flex: none;
    display: grid; place-items: center;
    border-radius: 13px;
    color: #eaf2ff;
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 60%, #0ea5e9 100%);
    box-shadow: 0 8px 22px rgba(59, 103, 246, 0.45),
                inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.side-brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 1.02rem; color: #eef1ff;
    line-height: 1.15; letter-spacing: -0.01em;
}
.side-brand-sub {
    font-size: 0.72rem; color: #7681a8;
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600;
}
.side-card {
    background: rgba(148, 163, 255, 0.06);
    border: 1px solid rgba(148, 163, 255, 0.14);
    border-radius: 16px;
    padding: 0.85rem 1rem;
    margin: 0.35rem 0 0.5rem;
}
.side-card-title {
    display: flex; align-items: center; gap: 0.45rem;
    color: #a7b0d8; font-size: 0.74rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.55rem;
}
.side-card-title svg { color: #8ab6ff; }
.side-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.22rem 0; font-size: 0.86rem; color: #a7b0d8;
}
.side-row b {
    color: #eef1ff; font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
}

/* ----- Hero ----- */
.hero { padding: 0.4rem 0 1.1rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.34rem 0.95rem;
    border-radius: 999px;
    border: 1px solid rgba(250, 178, 25, 0.45);
    background: rgba(35, 25, 8, 0.55);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: #ffd98a;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.hero-badge-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #fab219;
    box-shadow: 0 0 10px rgba(250, 178, 25, 0.9);
    animation: badge-pulse 2.4s ease-in-out infinite;
}
@keyframes badge-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.8); }
}
.hero-title {
    margin: 0.85rem 0 0.4rem;
    font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif;
    font-size: clamp(2.5rem, 5vw, 3.9rem);
    font-weight: 700; line-height: 1.06; letter-spacing: -0.025em;
    background: linear-gradient(93deg, #6ea8ff 0%, #9db4ff 42%, #5fe0ff 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
    filter: drop-shadow(0 3px 22px rgba(23, 59, 160, 0.55));
}
.hero-sub {
    margin: 0; max-width: 64ch;
    color: #c0c8ea; font-size: 1.04rem; line-height: 1.6;
    text-shadow: 0 1px 14px rgba(2, 5, 16, 0.85);
}
.hero-chips {
    display: flex; flex-wrap: wrap; gap: 0.6rem;
    margin-top: 1.15rem;
}
.chip {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.5rem 1rem;
    border-radius: 14px;
    background: rgba(13, 19, 46, 0.55);
    border: 1px solid rgba(148, 163, 255, 0.20);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 8px 28px rgba(2, 5, 16, 0.45),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
}
.chip svg { color: #8ab6ff; flex: none; }
.chip-val {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 1.02rem; color: #eef1ff;
}
.chip-lab {
    font-size: 0.72rem; font-weight: 600; color: #8b95bd;
    text-transform: uppercase; letter-spacing: 0.07em;
}

/* ----- Note « données fictives » (remplace l'alerte jaune) ----- */
.demo-note {
    display: flex; gap: 0.7rem; align-items: flex-start;
    padding: 0.8rem 1.05rem;
    border-radius: 16px;
    background: rgba(35, 25, 8, 0.45);
    border: 1px solid rgba(250, 178, 25, 0.30);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #d9c9a0; font-size: 0.86rem; line-height: 1.5;
    margin: 0.2rem 0 0.9rem;
}
.demo-note svg { color: #fab219; flex: none; margin-top: 2px; }
.demo-note b { color: #ffd98a; }

/* ----- Titres de section du formulaire ----- */
.sec {
    display: flex; align-items: center; gap: 0.7rem;
    margin: 0.35rem 0 0.25rem;
}
.sec-ic {
    width: 34px; height: 34px; flex: none;
    display: grid; place-items: center;
    border-radius: 10px;
    color: #9db9ff;
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.32), rgba(14, 165, 233, 0.18));
    border: 1px solid rgba(148, 163, 255, 0.25);
}
.sec-t {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 1.04rem; color: #eef1ff;
    letter-spacing: -0.01em;
}
.sec-s { font-size: 0.78rem; color: #7681a8; margin-top: 1px; }
.sec-rule {
    height: 1px; flex: 1;
    background: linear-gradient(90deg, rgba(148, 163, 255, 0.22), transparent);
}

/* ----- Formulaire principal : grande carte de verre ----- */
[data-testid="stMain"] [data-testid="stForm"] {
    background: rgba(10, 15, 38, 0.58);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(148, 163, 255, 0.16);
    border-radius: 22px;
    padding: 1.5rem 1.6rem 1.7rem;
    box-shadow: 0 18px 50px rgba(2, 5, 16, 0.55),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
[data-testid="stWidgetLabel"] p {
    color: #a7b0d8 !important;
    font-size: 0.84rem; font-weight: 500;
}

/* ----- Cartes de verre : métriques natives ----- */
[data-testid="stMetric"] {
    background: rgba(13, 19, 46, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 255, 0.14);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: 0 10px 34px rgba(3, 6, 20, 0.45);
}
[data-testid="stMetricLabel"] p { color: #a7b0d8 !important; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #eef1ff; }

/* ----- Bandeau KPI premium ----- */
.kpi-band {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.9rem;
    margin: 0.3rem 0 0.4rem;
}
.kpi {
    position: relative;
    background: rgba(11, 16, 40, 0.60);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 255, 0.16);
    border-radius: 18px;
    padding: 1rem 1.15rem 0.95rem;
    overflow: hidden;
    box-shadow: 0 12px 36px rgba(2, 5, 16, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}
.kpi::before {
    content: ""; position: absolute; inset: 0 auto auto 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, rgba(124, 140, 248, 0.65), rgba(95, 224, 255, 0.25), transparent 75%);
}
.kpi:hover {
    transform: translateY(-3px);
    border-color: rgba(124, 140, 248, 0.45);
    box-shadow: 0 18px 46px rgba(53, 73, 190, 0.35);
}
.kpi-top {
    display: flex; align-items: center; gap: 0.55rem;
    margin-bottom: 0.55rem;
    min-height: 34px;
}
.kpi-ic {
    width: 30px; height: 30px; flex: none;
    display: grid; place-items: center;
    border-radius: 9px; color: #9db9ff;
    background: rgba(124, 140, 248, 0.14);
    border: 1px solid rgba(148, 163, 255, 0.22);
}
.kpi-lab {
    font-size: 0.75rem; font-weight: 600; color: #8b95bd;
    text-transform: uppercase; letter-spacing: 0.07em;
    line-height: 1.25;
}
.kpi-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.72rem; font-weight: 700; color: #eef1ff;
    line-height: 1.08; letter-spacing: -0.02em;
    white-space: nowrap;
}
.kpi-val small { font-size: 0.95rem; font-weight: 600; color: #a7b0d8; }
.kpi-delta {
    margin-top: 0.4rem; font-size: 0.78rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.3rem; color: #7681a8;
}
.kpi-delta.good { color: #5fd79a; }
.kpi-delta.bad  { color: #ff8a7a; }
.kpi-delta span { color: #7681a8; font-weight: 500; }

/* ----- Carte verdict premium ----- */
.verdict {
    background: rgba(10, 15, 38, 0.62);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(148, 163, 255, 0.18);
    border-radius: 22px;
    padding: 1.5rem 1.6rem 1.4rem;
    box-shadow: 0 18px 50px rgba(2, 5, 16, 0.55),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.verdict-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.32rem 0.85rem; border-radius: 999px;
    font-size: 0.76rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.verdict-badge svg { flex: none; }
.v-good     { background: rgba(12, 163, 12, 0.16);  border: 1px solid rgba(76, 217, 100, 0.45);  color: #7fe3a5; }
.v-warn     { background: rgba(250, 178, 25, 0.14); border: 1px solid rgba(250, 178, 25, 0.45);  color: #ffd98a; }
.v-serious  { background: rgba(236, 131, 90, 0.14); border: 1px solid rgba(236, 131, 90, 0.5);   color: #ffb391; }
.v-crit     { background: rgba(208, 59, 59, 0.16);  border: 1px solid rgba(240, 100, 100, 0.5);  color: #ff9d94; }
.verdict-prob {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.4rem; font-weight: 700; color: #eef1ff;
    line-height: 1; letter-spacing: -0.03em;
    margin: 0.75rem 0 0.1rem;
}
.verdict-prob small { font-size: 1.5rem; color: #a7b0d8; font-weight: 600; }
.verdict-cap {
    font-size: 0.78rem; color: #8b95bd; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.verdict-msg {
    margin: 0.8rem 0 0;
    color: #c0c8ea; font-size: 0.93rem; line-height: 1.55;
}
.verdict-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem;
    margin-top: 1.05rem; padding-top: 1rem;
    border-top: 1px solid rgba(148, 163, 255, 0.14);
}
.verdict-cell-lab {
    font-size: 0.7rem; color: #7681a8; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.verdict-cell-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.0rem; font-weight: 600; color: #eef1ff; margin-top: 2px;
}

/* ----- État vide (avant soumission) ----- */
.empty-state {
    background: rgba(10, 15, 38, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px dashed rgba(148, 163, 255, 0.30);
    border-radius: 22px;
    padding: 2rem 1.6rem;
    text-align: center;
}
.empty-state .sec-ic { margin: 0 auto 0.8rem; width: 46px; height: 46px; border-radius: 14px; }
.empty-state-t {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 1.1rem; color: #eef1ff;
}
.empty-state-s { color: #8b95bd; font-size: 0.88rem; margin-top: 0.3rem; line-height: 1.5; }

/* ----- Cartes de verre : graphiques, expandeurs, alertes, tableaux ----- */
[data-testid="stPlotlyChart"] {
    background: rgba(11, 16, 40, 0.58);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(148, 163, 255, 0.13);
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
    background: rgba(11, 16, 40, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 255, 0.13);
    border-radius: 16px;
}
[data-testid="stAlertContainer"], [data-testid="stAlert"] {
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
}
[data-testid="stDataFrame"] {
    background: rgba(11, 16, 40, 0.55);
    border: 1px solid rgba(148, 163, 255, 0.15);
    border-radius: 16px;
    overflow: hidden;
}

/* ----- Boutons : dégradé électrique + glow ----- */
.stButton > button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button, [data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #4f46e5 0%, #6d5ef1 55%, #3b82f6 100%) !important;
    color: #f5f7ff !important;
    border: 1px solid rgba(165, 180, 252, 0.35) !important;
    border-radius: 14px !important;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 30px rgba(79, 70, 229, 0.45);
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover, [data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    filter: brightness(1.12);
    box-shadow: 0 16px 42px rgba(99, 102, 241, 0.55);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0);
}

/* ----- Liens de page (sidebar) ----- */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 255, 0.14);
    background: rgba(148, 163, 255, 0.05);
    transition: background .2s ease, border-color .2s ease;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background: rgba(124, 140, 248, 0.16);
    border-color: rgba(124, 140, 248, 0.4);
}

/* ----- Onglets (react-aria, Streamlit >= 1.65) ----- */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.35rem;
    background: rgba(11, 16, 40, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 255, 0.14);
    border-radius: 16px;
    padding: 0.3rem;
    width: fit-content;
    border-bottom: 1px solid rgba(148, 163, 255, 0.14);
}
[data-testid="stTab"] {
    border-radius: 12px;
    padding: 0.4rem 1.1rem;
    transition: background .2s ease;
}
[data-testid="stTab"] p { color: #a7b0d8; transition: color .2s ease; }
[data-testid="stTab"]:hover { background: rgba(124, 140, 248, 0.14); }
[data-testid="stTab"]:hover p { color: #eef1ff; }
[data-testid="stTab"][aria-selected="true"] {
    background: linear-gradient(120deg, rgba(79, 70, 229, 0.6), rgba(59, 130, 246, 0.5));
    box-shadow: 0 4px 18px rgba(79, 70, 229, 0.35);
}
[data-testid="stTab"][aria-selected="true"] p { color: #ffffff; font-weight: 600; }
[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }

/* ----- Footer ----- */
.app-footer {
    margin-top: 2.6rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(148, 163, 255, 0.12);
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.5rem;
    color: #7681a8; font-size: 0.8rem;
}
.app-footer b { color: #a7b0d8; font-weight: 600; }

/* ----- Divers ----- */
hr { border-color: rgba(148, 163, 255, 0.14); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 255, 0.22);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 255, 0.38); }
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
"""


def inject_css() -> None:
    """Injecte le thème global (fond photo + scrim, verre, typo) dans la page."""
    st.markdown(_CSS.replace("__BG64__", _bg_data_uri()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Composants HTML partagés
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str, badges: list[tuple[str, str, str]] | None = None,
         badge: str = "Démo — données 100 % fictives") -> None:
    """Hero de page : badge démo, grand titre dégradé, sous-titre, chips de verre.

    ``badges`` : liste de tuples ``(icône, valeur, libellé)`` rendus en chips.
    """
    chips = ""
    if badges:
        chips = '<div class="hero-chips">' + "".join(
            f'<span class="chip">{icon(name, 17)}'
            f'<span class="chip-val">{val}</span>'
            f'<span class="chip-lab">{lab}</span></span>'
            for name, val, lab in badges
        ) + "</div>"
    st.markdown(
        f"""
        <div class="hero">
          <span class="hero-badge"><span class="hero-badge-dot"></span>{badge}</span>
          <h1 class="hero-title">{title}</h1>
          <p class="hero-sub">{subtitle}</p>
          {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def demo_note(text_html: str) -> None:
    """Note discrète « données fictives » (remplace l'alerte jaune native)."""
    st.markdown(
        f'<div class="demo-note">{icon("shield", 18)}<div>{text_html}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(icon_name: str, title: str, sub: str = "") -> None:
    """Titre de section iconé pour le formulaire (Emprunteur, Crédit, …)."""
    sub_html = f'<div class="sec-s">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="sec">
          <span class="sec-ic">{icon(icon_name, 18)}</span>
          <div><div class="sec-t">{title}</div>{sub_html}</div>
          <span class="sec-rule"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_band(items: list[dict]) -> None:
    """Bandeau de cartes KPI premium.

    Chaque item : ``{icon, label, value, unit?, delta?, good?, sub?}`` —
    ``delta`` (str) coloré selon ``good`` (True vert / False rouge / None neutre),
    ``sub`` = complément discret après le delta.
    """
    cards = []
    for it in items:
        unit = f' <small>{it["unit"]}</small>' if it.get("unit") else ""
        delta_html = ""
        if it.get("delta") is not None:
            cls = "good" if it.get("good") else ("bad" if it.get("good") is False else "")
            arrow = "▲" if str(it["delta"]).startswith("+") else ("▼" if str(it["delta"]).startswith("−") else "•")
            sub = f' <span>{it["sub"]}</span>' if it.get("sub") else ""
            delta_html = f'<div class="kpi-delta {cls}">{arrow} {it["delta"]}{sub}</div>'
        elif it.get("sub"):
            delta_html = f'<div class="kpi-delta"><span>{it["sub"]}</span></div>'
        cards.append(
            f'<div class="kpi">'
            f'<div class="kpi-top"><span class="kpi-ic">{icon(it["icon"], 16)}</span>'
            f'<span class="kpi-lab">{it["label"]}</span></div>'
            f'<div class="kpi-val">{it["value"]}{unit}</div>'
            f'{delta_html}</div>'
        )
    st.markdown(f'<div class="kpi-band">{"".join(cards)}</div>', unsafe_allow_html=True)


def sidebar_brand(sub: str = "Démo · Machine Learning") -> None:
    """Bloc de marque en tête de sidebar."""
    st.sidebar.markdown(
        f"""
        <div class="side-brand">
          <span class="side-brand-logo">{icon("bank", 22)}</span>
          <div>
            <div class="side-brand-name">Credit Scoring</div>
            <div class="side-brand-sub">{sub}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_page_link(page: str, label: str, icon_str: str) -> None:
    """Lien de navigation sidebar, sans échec quand la page est exécutée seule
    (AppTest ou ``streamlit run`` direct sur un fichier de ``pages/``)."""
    try:
        st.sidebar.page_link(page, label=label, icon=icon_str)
    except Exception:  # page hors registre multipage : lien non rendu
        pass


def footer(left: str, right: str = "") -> None:
    """Footer discret en bas de page."""
    st.markdown(
        f'<div class="app-footer"><span>{left}</span><span>{right}</span></div>',
        unsafe_allow_html=True,
    )
