import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import RobustScaler
import streamlit as st
import plotly.graph_objects as go
import openpyxl
import joblib
import time
import plotly
st.set_page_config(layout="wide")
#Création du dataframe
data=openpyxl.load_workbook("BD finale.xlsx")
datas=data.active
donnees = []
for ligne in datas.iter_rows(values_only=True):
    donnees.append(list(ligne))
import pandas as pd
df = pd.DataFrame(donnees)

en_tetes = donnees[0]
donnees = donnees[1:]
df = pd.DataFrame(donnees, columns=en_tetes)
df.set_index(en_tetes[0], inplace=True)  # En faire l'index
df['AGE'] = df['AGE'].astype(int)
df['MNTPRT'] = df['MNTPRT'].astype(int)
df["MARGE"].replace('null', 0, inplace=True)
df["MARGE"]=df["MARGE"].astype(float)
# Convertir la colonne "Date de déblocage" en format de date
df['DATDEP'] = pd.to_datetime(df['DATDEP'], errors='coerce' )
df["DERNDAT"] = pd.to_datetime(df["DERNDAT"], errors='coerce')


#Définition des fonctions


# Appliquer le RobustScaler aux nouvelles données
def apply_robust_scaler(value, var):
    params = {
    'median': df[var].median(),
    'q1': df[var].quantile(0.25),
    'q3': df[var].quantile(0.75)
}
    scaled_value = (value - params['median']) / (params['q3'] - params['q1'])
    return scaled_value

def adjust_categorical_values1(new_values, saved_categories):
    adjusted_values = [1 if value == new_values else 0 for value in saved_categories]
    return adjusted_values

def reverse_onehot_encode1(new_values, categories_mapping):
    reversed_values = []
    for variable, categories in categories_mapping.items():
        adjusted_values = adjust_categorical_values1(new_values[variable], categories)
        reversed_values.extend(adjusted_values)
    return reversed_values

def catégorielle(values_f):
    categories_mapping = {
        'SEXE': ['homme'],
        'SITUATION_MAT': ['célibataire', 'marié(e)', 'divorcé(e)','veuf(ve)'],
        'ACTILIB': ['ADMINISTRATIF','AGRO ALIMENTAIRE','ASSURANCE','BANQUE','COMMERCIAL','DIPLOMATIE','DIRECTION GENERALE','DIVERS','ETUDES/RECH./DEVELOP.','FINANCE','INFORMATIQUE, ORGANIS.','JURIDIQUE','MARKETING, PUBLICITE','PRODUCTION','PROFESSIONS LIBERALES','RESSOURCES HUMAINES'],
        'LIBELLE': ['BRIDGE PRET RELAIS','CCT AUTRES CRD','CCT CONSO','CCT CONSO BONNE GAMME','CCT CONSO PER AUTRE','CCT HORS PP','CCT RESTRUCTURES','CCT SCOLAIRE','CMT AUTRES','CMT CONSO','CMT CONSO BONNE GAMME','CMT CONSO PERSO','CMT HAB BONNE GAMME','CMT HAB PATRIMONIALE','CMT HORS PP','CMT RESTRUCTURES'],
        'AGENCELIB': ['AGENCE - ADJAME','AGENCE - AG PRINCIPALE','AGENCE - COCODY','AGENCE - II PLATEAUX 8IEME TRANCHE','AGENCE - MARCORY RESIDENTIEL','AGENCE - PLATEAU SEEN HOTEL','AGENCE - RIVIERA 3','AGENCE - RIVIERA GOLF','AGENCE - SAN PEDRO','AGENCE - TREICHVILLE ZONE 3','AGENCE - ZONE 4 DR BLANCHARD','AGENCE- 2 PLATEAUX RUE DES JARDINS','AGENCE- DEUX PLATEAUX LATRILLE','AGENCE-TREICHVILLE NANAN YAMOUSSO']
    }
    return reverse_onehot_encode1(values_f, categories_mapping)


def adjust_categorical_values2(new_values, saved_categories):
    adjusted_values = [1 if value in new_values else 0 for value in saved_categories]
    return adjusted_values

def reverse_onehot_encode2(new_values, categories_mapping):
    reversed_values = []
    for variable, categories in categories_mapping.items():
        adjusted_values = adjust_categorical_values2(new_values[variable], categories)
        reversed_values.extend(adjusted_values)
    return reversed_values
categories_mapping_garanties = {
        'domiciliation des revenus': 0,
        'CASH COLL': 1,
        'DAT': 2,
        'garantie hypothécaire': 3,
        'garantie progressive': 4,
        'assurance multirisque': 5,
        'Billet à Ordre': 6,
        'clean': 7
    }
def modif_garanties(new_values_garanties):
    # Exemple d'utilisation pour la variable catégorielle 'garanties'


    # Ajuster les valeurs catégorielles pour 'garanties'
    adjusted_values_garanties = reverse_onehot_encode2({'garanties': new_values_garanties}, {'garanties': categories_mapping_garanties})

    return adjusted_values_garanties

def map_age_interval_vector(age_value):
    age_intervals = {
        'Age_20-29': [20, 29],
        'Age_30-39': [30, 39],
        'Age_40-49': [40, 49],
        'Age_50-59': [50, 59],
        'Age_60-69': [60, 69],
        'Age_70+': [70, float('inf')]
    }

    vector = np.zeros(len(age_intervals), dtype=int)  # Initialisez un vecteur de zéros

    for i, (interval, bounds) in enumerate(age_intervals.items()):
        if bounds[0] <= age_value <= bounds[1]:
            vector[i] = 1  # Affectez 1 si la valeur appartient à l'intervalle

    return vector.tolist()


model = joblib.load("best_model1.pkl")

def prédire(x):
    pred = model.predict(x)
    proba = model.predict_proba(x)
    proba=np.round(proba*100, 4)
    return pred,proba
# Créer une fonction pour l'application Streamlit
def main():


    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url(https://teyliom.com/wp-content/uploads/2021/03/BBGCI-TOF.jpg);
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: no-fixed;
    height: 100vh;
    margin: 0;
    display: flex;

    }}
    .ribbon {{
        background-color: #000;
        color: #fff;
        text-align: ;
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
    </style>
    """


    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;width:100%;"><h1 style="color:white;background-color:black;border:red;border-style:solid;border-radius:5px; padding: 10px;">APPLICATION DE CREDIT SCORING BANCAIRE</h1></div>', unsafe_allow_html=True)
    st.write("\n")
    st.write("\n")

    #st.markdown('<div class="ribbon">APPLICATION DE CREDIT SCORING BANCAIRE</div>', unsafe_allow_html=True)
    st.sidebar.image('logo.jpg', use_column_width='always')
    st.sidebar.title("Informations sur le client")
    duration = st.sidebar.slider("Durée du Remboursement (en mois)", min_value=1, max_value=120, value=60)
    amount = st.sidebar.number_input("Montant du Prêt (en FCFA)", min_value=0)
    margin = st.sidebar.number_input("Taux d'intérêt (en %)", min_value=0)
    sex = st.sidebar.selectbox("Sexe", ['homme', 'femme'])
    marital_status = st.sidebar.selectbox("Situation Matrimoniale", ['célibataire', 'marié(e)', 'divorcé(e)', 'veuf(ve)'], )
    job = st.sidebar.selectbox("Activité", ['ADMINISTRATIF', 'AGRO ALIMENTAIRE', 'ASSURANCE', 'BANQUE', 'COMMERCIAL', 'DIPLOMATIE', 'DIRECTION GENERALE', 'DIVERS', 'ETUDES/RECH./DEVELOP.', 'FINANCE', 'INFORMATIQUE, ORGANIS.', 'JURIDIQUE', 'MARKETING, PUBLICITE', 'PRODUCTION', 'PROFESSIONS LIBERALES', 'RESSOURCES HUMAINES'])
    label = st.sidebar.selectbox("Type de prêt", ['BRIDGE PRET RELAIS', 'CCT AUTRES CRD', 'CCT CONSO', 'CCT CONSO BONNE GAMME', 'CCT CONSO PER AUTRE', 'CCT HORS PP', 'CCT RESTRUCTURES', 'CCT SCOLAIRE', 'CMT AUTRES', 'CMT CONSO', 'CMT CONSO BONNE GAMME', 'CMT CONSO PERSO', 'CMT HAB BONNE GAMME', 'CMT HAB PATRIMONIALE', 'CMT HORS PP', 'CMT RESTRUCTURES'])
    agency = st.sidebar.selectbox("Agence", ['AGENCE - ADJAME', 'AGENCE - AG PRINCIPALE', 'AGENCE - COCODY', 'AGENCE - II PLATEAUX 8IEME TRANCHE', 'AGENCE - MARCORY RESIDENTIEL', 'AGENCE - PLATEAU SEEN HOTEL', 'AGENCE - RIVIERA 3', 'AGENCE - RIVIERA GOLF', 'AGENCE - SAN PEDRO', 'AGENCE - TREICHVILLE ZONE 3', 'AGENCE - ZONE 4 DR BLANCHARD', 'AGENCE- 2 PLATEAUX RUE DES JARDINS', 'AGENCE- DEUX PLATEAUX LATRILLE', 'AGENCE-TREICHVILLE NANAN YAMOUSSO'])
    age = st.sidebar.number_input("Age (en années)", min_value=18)
    garantie = st.sidebar.multiselect("Sélectionnez les garanties :", list(categories_mapping_garanties.keys()))
    month = st.sidebar.number_input("Mois d'octroi du crédit", min_value=1, max_value=12)


    # Créer un dataframe temporaire pour stocker les valeurs entrées par l'utilisateur
    l=[apply_robust_scaler(duration,'DUR_REMB'), apply_robust_scaler(amount, 'MNTPRT'), margin]
    l.extend(catégorielle({'SEXE': sex, 'SITUATION_MAT': marital_status, 'ACTILIB': job, 'LIBELLE': label, 'AGENCELIB': agency}))
    l.extend(map_age_interval_vector(age))
    l.extend(modif_garanties(garantie))
    l.extend([month])
    user_input_df = pd.DataFrame(l).T

    # Afficher le dataframe résultant
    #st.subheader("Données d'entrée utilisateur :")
    #st.write(user_input_df)
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
        jauge(p)
        col1, col2, col3 = st.columns([2.85,4.95,2.20])
        with col1:
            st.write(' ')
            
        with col2:
            
            st.plotly_chart(jauge(p))
        with col3:
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
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
            st.plotly_chart(jauge(0))
        with col6:
            st.write(' ')
        
        
        
# Exécuter l'application
if __name__ == '__main__':
    main()

