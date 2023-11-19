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
import openpyxl
import joblib
import time

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
def apply_robust_scaler(value):
    params = {
    'median': df['DUR_REMB'].median(),
    'q1': df['DUR_REMB'].quantile(0.25),
    'q3': df['DUR_REMB'].quantile(0.75)
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
        'SEXE': ['homme', 'femme'],
        'SITUATION_MAT': ['célibataire', 'marié', 'divorcé'],
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
    proba=np.round(proba*100, 2)
    return pred,proba
# Créer une fonction pour l'application Streamlit
def main():

    @st.cache_data
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()



    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url(https://www.sygnum.com/wp-content/uploads/2022/11/AdobeStock_453778834-scaled.jpeg);
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
        text-align: center;
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
    st.markdown('<div class="ribbon">APPLICATION DE CREDIT SCORING BANCAIRE</div>', unsafe_allow_html=True)
    st.sidebar.title("Entrée Utilisateur")
    duration = st.sidebar.slider("Durée du Remboursement (mois)", min_value=1, max_value=120, value=60)
    amount = st.sidebar.number_input("Montant du Prêt", min_value=0)
    margin = st.sidebar.number_input("Marge", min_value=0)
    sex = st.sidebar.selectbox("Sexe", ['homme', 'femme'])
    marital_status = st.sidebar.selectbox("Situation Matrimoniale", ['célibataire', 'marié', 'divorcé'])
    job = st.sidebar.selectbox("Activité", ['ADMINISTRATIF', 'AGRO ALIMENTAIRE', 'ASSURANCE', 'BANQUE', 'COMMERCIAL', 'DIPLOMATIE', 'DIRECTION GENERALE', 'DIVERS', 'ETUDES/RECH./DEVELOP.', 'FINANCE', 'INFORMATIQUE, ORGANIS.', 'JURIDIQUE', 'MARKETING, PUBLICITE', 'PRODUCTION', 'PROFESSIONS LIBERALES', 'RESSOURCES HUMAINES'])
    label = st.sidebar.selectbox("Libellé", ['BRIDGE PRET RELAIS', 'CCT AUTRES CRD', 'CCT CONSO', 'CCT CONSO BONNE GAMME', 'CCT CONSO PER AUTRE', 'CCT HORS PP', 'CCT RESTRUCTURES', 'CCT SCOLAIRE', 'CMT AUTRES', 'CMT CONSO', 'CMT CONSO BONNE GAMME', 'CMT CONSO PERSO', 'CMT HAB BONNE GAMME', 'CMT HAB PATRIMONIALE', 'CMT HORS PP', 'CMT RESTRUCTURES'])
    agency = st.sidebar.selectbox("Agence", ['AGENCE - ADJAME', 'AGENCE - AG PRINCIPALE', 'AGENCE - COCODY', 'AGENCE - II PLATEAUX 8IEME TRANCHE', 'AGENCE - MARCORY RESIDENTIEL', 'AGENCE - PLATEAU SEEN HOTEL', 'AGENCE - RIVIERA 3', 'AGENCE - RIVIERA GOLF', 'AGENCE - SAN PEDRO', 'AGENCE - TREICHVILLE ZONE 3', 'AGENCE - ZONE 4 DR BLANCHARD', 'AGENCE- 2 PLATEAUX RUE DES JARDINS', 'AGENCE- DEUX PLATEAUX LATRILLE', 'AGENCE-TREICHVILLE NANAN YAMOUSSO'])
    age = st.sidebar.number_input("Age", min_value=0)
    garantie = st.sidebar.multiselect("Sélectionnez les garanties :", list(categories_mapping_garanties.keys()))
    month = st.sidebar.number_input("Mois_d'octroi_du crédit", min_value=1, max_value=12)
    year = st.sidebar.number_input("Année d'octroi du crédit", min_value=2006, max_value=2050)


    # Créer un dataframe temporaire pour stocker les valeurs entrées par l'utilisateur
    l=[duration, apply_robust_scaler(amount), margin]
    l.extend(catégorielle({'SEXE': sex, 'SITUATION_MAT': marital_status, 'ACTILIB': job, 'LIBELLE': label, 'AGENCELIB': agency}))
    l.extend(map_age_interval_vector(age))
    l.extend(modif_garanties(garantie))
    l.extend([month,year])
    user_input_df = pd.DataFrame(l).T

    # Afficher le dataframe résultant
    #st.subheader("Données d'entrée utilisateur :")
    #st.write(user_input_df)
    if st.sidebar.button("Prédire"):
        predict, probability=prédire(user_input_df)
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success('Prédiction effectuée!')
        p=pd.DataFrame(probability)[1][0]
        st.subheader(f"la probabilité de défaut de paiement est de:{p}%")
        


# Exécuter l'application
if __name__ == '__main__':
    main()

