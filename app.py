import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib
import time
from pathlib import Path

from features import GARANTIES, build_features
from generate_synthetic_data import AGENCES, SECTEURS, SITUATIONS, TYPES_PRET

st.set_page_config(layout="wide")

ROOT = Path(__file__).parent

# Données 100 % fictives (générées par generate_synthetic_data.py) et modèle
# ré-entraîné sur ces données (train_model.py) : Pipeline scikit-learn complet,
# qui porte la mise à l'échelle robuste et l'encodage one-hot d'origine.
categories_mapping_garanties = {g: i for i, g in enumerate(GARANTIES)}


@st.cache_resource
def charger_modele():
    return joblib.load(ROOT / "model" / "model.pkl")


model = charger_modele()


def prédire(x):
    pred = model.predict(x)
    proba = model.predict_proba(x)
    proba = np.round(proba * 100, 4)
    return pred, proba


# Créer une fonction pour l'application Streamlit
def main():


    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"] {{
    background-image: url(https://teyliom.com/wp-content/uploads/2021/03/BBGCI-TOF.jpg);
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: scroll;
    height: 100vh;
    margin: 0;
    display: flex;

    }}
    .ribbon {{
        background-color: #000;
        color: #fff;
        padding: 10px;
        font-size: 24px;
        font-family: 'Arial', sans-serif;  /* Choisir une belle police */
        font-weight: bold;  /* Mettre le texte en gras */
        border: 2px solid #ff0000;  /* Bordures rouges */
        border-radius: 10px;  /* Coins arrondis */
        margin-top: -50px;  /* Ajuster la position vers le haut */
        position: relative;
        z-index: 1;  /* S'assurer que le ruban est au-dessus du contenu */
    }}
    [data-testid="stSidebar"] {{
        background-color: #000 !important;  /* Fond noir */
        border: 2px solid #ff0000 !important;  /* Bordure rouge */
        border-radius: 10px;  /* Coins arrondis */
        margin-top: -30px;  /* Ajuster la position vers le haut */
        position: relative;
        z-index: 1;  /* S'assurer que la barre latérale est au-dessus du contenu */
        padding: 10px;
    }}
        [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
        color: white;
    }}


    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    .espace-verdict {{
        height: 283px;
    }}
    /* Grands écrans : jauge à sa largeur d'origine (600 px) */
    @media (min-width: 1280px) {{
        .st-key-jauge {{
            min-width: 600px;
        }}
    }}
    @media (min-width: 769px) {{
        [data-testid="stSidebar"][aria-expanded="true"] {{
            width: 336px !important;
            height: 100vh !important;
        }}
    }}
    .mention-demo {{
        text-align: center;
        margin: 6px 0 0 0;
    }}
    .mention-demo span {{
        background-color: rgba(0, 0, 0, 0.6);
        color: #ddd;
        font-size: 0.8rem;
        padding: 2px 10px;
        border-radius: 5px;
    }}

    /* Écrans moyens (tablettes) */
    @media (max-width: 1279px) {{
        .st-key-jauge text.title {{
            font-size: 22px !important;
        }}
    }}
    @media (max-width: 1024px) {{
        h1.titre-app {{
            font-size: 1.9rem !important;
        }}
    }}
    /* Téléphones */
    @media (max-width: 768px) {{
        [data-testid="stSidebar"] {{
            margin-top: 0;
        }}
        h1.titre-app {{
            font-size: 1.6rem !important;
            padding: 8px !important;
        }}
        [data-testid="stPlotlyChart"] text.title {{
            font-size: 17px !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
    /* Colonnes empilées (petits écrans) */
    @media (max-width: 640px) {{
        .espace-verdict {{
            height: 0;
        }}
        h1.titre-app {{
            font-size: 1.3rem !important;
        }}
    }}
    </style>
    """


    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;width:100%;"><h1 class="titre-app" style="color:white;background-color:black;border:red;border-style:solid;border-radius:5px; padding: 10px;">APPLICATION DE CREDIT SCORING BANCAIRE</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="mention-demo"><span>Données fictives — démonstration</span></p>', unsafe_allow_html=True)
    st.write("\n")

    st.sidebar.title("Informations sur le client")
    duration = st.sidebar.slider("Durée du Remboursement (en mois)", min_value=1, max_value=120, value=60)
    amount = st.sidebar.number_input("Montant du Prêt (en FCFA)", min_value=0)
    margin = st.sidebar.number_input("Taux d'intérêt (en %)", min_value=0)
    sex = st.sidebar.selectbox("Sexe", ['homme', 'femme'])
    marital_status = st.sidebar.selectbox("Situation Matrimoniale", SITUATIONS)
    job = st.sidebar.selectbox("Activité", SECTEURS)
    label = st.sidebar.selectbox("Type de prêt", TYPES_PRET)
    agency = st.sidebar.selectbox("Agence", AGENCES)
    age = st.sidebar.number_input("Age (en années)", min_value=18)
    garantie = st.sidebar.multiselect("Sélectionnez les garanties :", list(categories_mapping_garanties.keys()))
    month = st.sidebar.number_input("Mois d'octroi du crédit", min_value=1, max_value=12)


    # Créer un dataframe temporaire pour stocker les valeurs entrées par l'utilisateur
    # (le pipeline du modèle applique lui-même le RobustScaler et l'encodage one-hot)
    user_input_df = build_features(pd.DataFrame([{
        "DUREE DE REMBOURSEMENT": duration,
        "MONTANT SOLLICITE": amount,
        "TAUX D'INTERET": margin,
        "SEXE": sex,
        "SITUATION MATRIMONIALE": marital_status,
        "SECTEUR D'ACTIVITE": job,
        "TYPE DE PRÊT": label,
        "AGENCE": agency,
        "AGE": age,
        "GARANTIES": "; ".join(garantie),
        "MOIS D'OCTROI": month,
    }]))

    def jauge(prob):
        fig_jauge = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            # Customer scoring in % df_dashboard['SCORE_CLIENT_%']
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'JAUGE DE LA PROBABILITE DE DEFAUT', 'font': {'size': 30}},
            # Scoring of the 10 neighbourgs - test set
            # df_dashboard['SCORE_10_VOISINS_MEAN_TEST']
            delta={'reference': 70,
                'increasing': {'color': 'Crimson'},
                'decreasing': {'color': 'Green'}},
            gauge={'axis': {'range': [None, 100],
                            'tickwidth': 3,
                            'tickcolor': 'darkblue'},
                'bar': {'color': 'white', 'thickness': 0.25},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': 'gray',
                'steps': [{'range': [0, 25], 'color': 'Green'},
                            {'range': [25, 49.49], 'color': 'LimeGreen'},
                            {'range': [49.5, 50.5], 'color': 'red'},
                            {'range': [50.51, 69.99], 'color': 'Orange'},
                            {'range': [70, 100], 'color': 'Crimson'}],
                'threshold': {'line': {'color': 'white', 'width': 10},
                                'thickness': 0.8,
                                # Customer scoring in %
                                # df_dashboard['SCORE_CLIENT_%']
                                'value':prob}}))


        fig_jauge.update_layout(paper_bgcolor='rgba(0, 0, 0, 0.3)',
                                plot_bgcolor='rgba(0, 0, 0, 0.3)',
                                height=500, width=600,
                                font={'color': 'white', 'family': 'Arial'},
                                margin=dict(l=0, r=0, b=0, t=0, pad=0),
                                showlegend=False,
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                                shapes=[
                                    dict(
                                        type='rect',
                                        xref='paper',
                                        yref='paper',
                                        x0=0,
                                        y0=0,
                                        x1=1,
                                        y1=1,
                                        fillcolor='rgba(0, 0, 0, 0.3)',
                                        opacity=1,
                                        layer='below',
                                        line=dict(width=4, color='red'),
                                    )
                                ]
                                )
        return fig_jauge

    if st.sidebar.button("Prédire"):
        predict, probability=prédire(user_input_df)
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in [0,33,66]:
            time.sleep(0.01)
            my_bar.progress(percent_complete + 33, text=progress_text)
            time.sleep(1)
            my_bar.empty()



        p=pd.DataFrame(probability)[1][0]
        st.subheader(f"la probabilité de défaut de paiement est de:{p}%")
        col1, col2, col3 = st.columns([2.85,4.95,2.20])
        with col1:
            st.write(' ')

        with col2:

            st.plotly_chart(jauge(p), width="content", key="jauge")
        with col3:
            st.markdown('<div class="espace-verdict"></div>', unsafe_allow_html=True)
            if 0 <= p < 25:
                score_text = 'Crédit score : EXCELLENT'
                st.success(score_text)
            elif 25 <= p < 50:
                score_text = 'Crédit score : BON'
                st.success(score_text)
            elif 50 <= p < 70:
                score_text = 'Crédit score : MOYEN'
                st.warning(score_text)
            else:
                score_text = 'Crédit score : ÉLEVÉ \n (crédit potentiellement risqué!)'
                st.error(score_text)
    else:
        #st.error("Appuyez sur le bouton 'prédire' pour effectuer votre prédiction")
        col4, col5, col6 = st.columns([2.75,7,0.25])
        with col4:
            st.write(' ')
        with col5:
            st.plotly_chart(jauge(0), width="content")
        with col6:
            st.write(' ')



# Exécuter l'application
if __name__ == '__main__':
    main()
