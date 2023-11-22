import streamlit as st
import pandas as pd
import plotly.express as px
import openpyxl

# Chargement des données
data = openpyxl.load_workbook("BD finale.xlsx")
datas = data.active
donnees = []
for ligne in datas.iter_rows(values_only=True):
    donnees.append(list(ligne))
en_tetes = donnees[0]
donnees = donnees[1:]
df = pd.DataFrame(donnees, columns=en_tetes)
df.set_index(en_tetes[0], inplace=True)
df['AGE'] = df['AGE'].astype(int)
df['MNTPRT'] = df['MNTPRT'].astype(int)
df["MARGE"].replace('null', 0, inplace=True)
df["MARGE"] = df["MARGE"].astype(float)
df['DATDEP'] = pd.to_datetime(df['DATDEP'], errors='coerce' )
df["DERNDAT"] = pd.to_datetime(df["DERNDAT"], errors='coerce')
df.drop(columns=['NOM'], inplace=True)
d=df.groupby('AGENCELIB')['STATUT'].sum()
df['STATUT']=df['STATUT'].astype(str)
# Thème des graphiques
df['AGENCELIB']=df['AGENCELIB'].replace('AGENCE- DEUX PLATEAUX LATRILLE','AGENCE- 2 PLATEAUX RUE DES JARDINS' )


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
st.markdown('<div style="text-align:center;"><h1 style="color:white;background-color:black;border:red;border-style:solid;border-radius:5px;">TABLEAU DE BORD INTERACTIF</h1></div>', unsafe_allow_html=True)

colors = px.colors.sequential.Rainbow_r
# Section des graphiques sommaires
st.markdown(page_bg_img, unsafe_allow_html=True)



# Histogramme et Camembert sur la même ligne
cam, hist = st.columns([8,2],gap='large')

with cam:
    st.sidebar.subheader("CAMEMBERT")
    selected_categorical_variable_p = st.sidebar.selectbox("***Sélectionnez une variable catégorielle pour le camembert:***", ['SEXE', 'SITUATION_MAT', 'ACTILIB', 'PROFESSION', 'AGENCELIB','RM', 'LIBELLE','STATUT'])
    category_counts = df[selected_categorical_variable_p].value_counts()
    fig_pie = px.pie(names=category_counts.index, values=category_counts.values, title=f"Répartition de la variable {selected_categorical_variable_p}",color_discrete_sequence=colors)
    fig_pie.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0.5)','paper_bgcolor': 'rgba(0, 0, 0, 0.5)',})
    st.plotly_chart(fig_pie, use_container_width=True)
with hist:
    st.sidebar.subheader("HISTOGRAMME")
    selected_categorical_variable = st.sidebar.selectbox("***Sélectionnez la variable catégorielle pour l'histogramme :***", ['SEXE', 'SITUATION_MAT', 'ACTILIB', 'PROFESSION', 'AGENCELIB','RM', 'LIBELLE','STATUT'])
    fig_histogram = px.histogram(data, x=df[selected_categorical_variable], color=df[selected_categorical_variable],title=f"Histogramme de {selected_categorical_variable}",color_discrete_sequence=colors)
    fig_histogram.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0.5)','paper_bgcolor': 'rgba(0, 0, 0, 0.5)',})
    st.plotly_chart(fig_histogram)



# Section des analyses croisées
st.sidebar.subheader("ANALYSE CROISEE ENTRE VARIABLES NUMERIQUES")
int_columns = df.select_dtypes(include="int").columns
float_columns = df.select_dtypes(include="float").columns
selected_variable_3 = st.sidebar.selectbox("***Variable 1 :***", int_columns.union(float_columns))
selected_variable_4 = st.sidebar.selectbox("***Variable 2 :***", int_columns.union(float_columns))

st.sidebar.subheader("ANALYSE CROISEE ENTRE VARIABLES CATEGORIELLES")
selected_variable_1 = st.sidebar.selectbox("***Variable 1 :***", ['SEXE', 'SITUATION_MAT', 'ACTILIB', 'PROFESSION', 'AGENCELIB','RM', 'LIBELLE','STATUT'])
selected_variable_2 = st.sidebar.selectbox("***Variable 2 :***", ['SEXE', 'SITUATION_MAT', 'ACTILIB', 'PROFESSION', 'AGENCELIB','RM', 'LIBELLE','STATUT'])

def barmode_selected(t):
    if t =='empilé':
        a='relative'  
    else: 
        a='group'
    return a

quant,qual=st.columns([8,2],gap='large')
type_graph=st.sidebar.radio("Choisissez le type de graphique", ['empilé','étalé'])

with quant:
    fig_scatter_matrix = px.scatter(df, x=selected_variable_3, y=selected_variable_4,color_discrete_sequence=colors)
    fig_scatter_matrix.update_layout(title=f'Nuage de points entre {selected_variable_3} et {selected_variable_4}')
    fig_scatter_matrix.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0.5)','paper_bgcolor': 'rgba(0, 0, 0, 0.5)',})
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)
with qual:
    fig_croisé = px.bar(df, x=selected_variable_1, color=selected_variable_2,barmode=barmode_selected(type_graph),color_discrete_sequence= colors)
    fig_croisé.update_layout(title=f'Graphique en barres groupées - {selected_variable_1 } vs {selected_variable_2 }')
    fig_croisé.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0.5)','paper_bgcolor': 'rgba(0, 0, 0, 0.5)',})
    st.plotly_chart(fig_croisé)

#Map_agencies
data = openpyxl.load_workbook(r"C:\Users\DELL I5\Desktop\Classeur2.xlsx")
datas = data.active
donnees = []
for ligne in datas.iter_rows(values_only=True):
    donnees.append(list(ligne))
en_tetes = donnees[0]
donnees = donnees[1:]
df2 = pd.DataFrame(donnees, columns=en_tetes)
df= pd.merge(df, df2, on='AGENCELIB')

occurences=df['AGENCELIB'].value_counts()
df['size']=occurences[df['AGENCELIB']].values


df['Nbre de crédits risqués par agence']=d[df['AGENCELIB']].values
df["Taux de défauts de paiement"]=df['Nbre de crédits risqués par agence']/df['size']
df["Taux de défauts de paiement en pct"]=df["Taux de défauts de paiement"].map(lambda x: f'{x:.2%}')
center_lat = 5.31908
center_lon = -4.01299
px.set_mapbox_access_token("pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw")

fig_map = px.scatter_mapbox(df, lat=df.latitude, lon=df.longitude, color="Taux de défauts de paiement", size="size",
                  color_continuous_scale=px.colors.diverging.RdYlGn_r, zoom=12,custom_data=[df['AGENCELIB'],df['size'],df['Taux de défauts de paiement en pct']],size_max=40)

fig_map.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0.5)','paper_bgcolor': 'rgba(0, 0, 0, 0.5)','height': 700,'width': 1300,'mapbox': {'center': {'lat': center_lat, 'lon': center_lon},'style': 'dark'}})
fig_map.update_traces(text=df['Taux de défauts de paiement en pct'], hovertemplate='<b>Agence</b>: %{customdata[0]}<br>'
                                                                                      '<b>Latitude</b>: %{lat:.4f}<br>'
                                                                                      '<b>Longitude</b>: %{lon:.4f}<br>'
                                                                                      '<b>Nbre de prêts</b>: %{customdata[1]}<br>'
                                                                                      '<b>Taux de prêts en défaut de paiement</b>: %{customdata[2]}',
                                                                                      hoverlabel=dict(font=dict(size=16, color='#ff0000')),

                                                                                      )

st.plotly_chart(fig_map)
