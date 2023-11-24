import streamlit as st
import pandas as pd
import plotly.express as px
import openpyxl

# Chargement des données
data = openpyxl.load_workbook("Copie de BD finale.xlsx")
datas = data.active
donnees = []
for ligne in datas.iter_rows(values_only=True):
    donnees.append(list(ligne))
en_tetes = donnees[0]
donnees = donnees[1:]
df = pd.DataFrame(donnees, columns=en_tetes)
df.set_index(en_tetes[0], inplace=True)
df['AGE'] = df['AGE'].astype(int)
df['MONTANT SOLLICITE'] = df['MONTANT SOLLICITE'].astype(int)
df["TAUX D'INTERET"].replace('null', 0, inplace=True)
df["TAUX D'INTERET"] = df["TAUX D'INTERET"].astype(float)
df['DATDEP'] = pd.to_datetime(df['DATDEP'], errors='coerce' )
df["DERNDAT"] = pd.to_datetime(df["DERNDAT"], errors='coerce')
df.drop(columns=['NOM'], inplace=True)
d=df.groupby('AGENCES')['STATUT'].sum()
# Thème des graphiques
df['AGENCES']=df['AGENCES'].replace('AGENCE- DEUX PLATEAUX LATRILLE','AGENCE- 2 PLATEAUX RUE DES JARDINS' )
df["MOIS D'OCTROI DU CREDIT"] = df["DATDEP"].dt.month

# Créez une nouvelle colonne pour l'année
df["ANNEE D'OCTROI DU CREDIT"] = df["DATDEP"].dt.year
df["MOIS D'OCTROI DU CREDIT"] = df["MOIS D'OCTROI DU CREDIT"].astype(str)

df["MOIS D'OCTROI DU CREDIT"]=df["MOIS D'OCTROI DU CREDIT"].map({"1":"Janvier","2":"Février","3":"Mars","4":"Avril","5":"Mai","6":"Juin","7":"Juillet","8":"Août","9":"Septembre","10":"Octobre","11":"Novembre","12":"Décembre"})
order_of_months = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
df["MOIS D'OCTROI DU CREDIT"] = pd.Categorical(df["MOIS D'OCTROI DU CREDIT"], categories=order_of_months, ordered=True)
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
st.markdown(
    """
    <style>
        body {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div style="text-align:center;width:182%;"><h1 style="color:white;background-color:black;border:red;border-style:solid;border-radius:5px;">TABLEAU DE BORD INTERACTIF</h1></div>', unsafe_allow_html=True)

colors = px.colors.sequential.Rainbow_r
# Section des graphiques sommaires
st.markdown(page_bg_img, unsafe_allow_html=True)



# Histogramme et Camembert sur la même ligne
cam, hist = st.columns([8,2],gap='medium')

with cam:
    st.sidebar.subheader("CAMEMBERT")
    selected_categorical_variable_p = st.sidebar.selectbox("***:gray[Sélectionnez une variable catégorielle pour le camembert:]***", ['SEXE', 'SITUATION MATRIMONIALE', "SECTEUR D'ACTIVITE", 'PROFESSION', 'AGENCES',"CHARGE D'AFFAIRES","TYPE DE PRÊT",'STATUT DU PRÊT'])
    category_counts = df[selected_categorical_variable_p].value_counts()
    fig_pie = px.pie(names=category_counts.index, values=category_counts.values, title=f"Répartition de la variable {selected_categorical_variable_p}",color_discrete_sequence=colors)
    fig_pie.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.25)
    st.plotly_chart(fig_pie, use_container_width=True)
with hist:
    st.sidebar.subheader("HISTOGRAMME")
    selected_categorical_variable = st.sidebar.selectbox("***:gray[Sélectionnez la variable catégorielle pour l'histogramme :]***", ['SEXE', 'SITUATION MATRIMONIALE', "SECTEUR D'ACTIVITE", 'PROFESSION', 'AGENCES',"CHARGE D'AFFAIRES","TYPE DE PRÊT","MOIS D'OCTROI DU CREDIT","ANNEE D'OCTROI DU CREDIT",'STATUT DU PRÊT'])
    fig_histogram = px.histogram(data, x=df[selected_categorical_variable], color=df[selected_categorical_variable],title=f"Histogramme de {selected_categorical_variable}",color_discrete_sequence=colors)
    fig_histogram.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.35)
    fig_histogram.update_traces( textfont_color='rgba(255, 255, 255, 1)')
    if selected_categorical_variable=="MOIS D'OCTROI DU CREDIT":
        fig_histogram.update_xaxes(categoryorder='array', categoryarray=order_of_months)
    st.plotly_chart(fig_histogram)



# Section des analyses croisées
st.sidebar.subheader("ANALYSE CROISEE ENTRE VARIABLES NUMERIQUES")
int_columns = df.select_dtypes(include="int").columns
float_columns = df.select_dtypes(include="float").columns
selected_variable_3 = st.sidebar.selectbox("***:gray[Variable 1 :]***", ['AGE','DUREE DE REMBOURSEMENT',"TAUX D'INTERÊT","MONTANT SOLLICITE"])
selected_variable_4 = st.sidebar.selectbox("***:gray[Variable 2 :]***", ['AGE','DUREE DE REMBOURSEMENT',"TAUX D'INTERÊT","MONTANT SOLLICITE"])

df["ANNEE D'OCTROI DU CREDIT"] = df["ANNEE D'OCTROI DU CREDIT"].astype(str)
order_of_years = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017','2018','2019','2020','2021','2022']
st.sidebar.subheader("ANALYSE CROISEE ENTRE VARIABLES CATEGORIELLES")
selected_variable_1 = st.sidebar.selectbox("***:gray[Variable 1 :]***", ['SEXE', 'SITUATION MATRIMONIALE', "SECTEUR D'ACTIVITE", 'PROFESSION', 'AGENCES',"CHARGE D'AFFAIRES","TYPE DE PRÊT","MOIS D'OCTROI DU CREDIT","ANNEE D'OCTROI DU CREDIT",'STATUT DU PRÊT'])
selected_variable_2 = st.sidebar.selectbox("***:gray[Variable 2 :]***", ['SEXE', 'SITUATION MATRIMONIALE', "SECTEUR D'ACTIVITE", 'PROFESSION', 'AGENCES',"CHARGE D'AFFAIRES","TYPE DE PRÊT","MOIS D'OCTROI DU CREDIT","ANNEE D'OCTROI DU CREDIT",'STATUT DU PRÊT'])

def barmode_selected(t):
    if t =='empilé':
        a='relative'  
    else: 
        a='group'
    return a

quant,qual=st.columns([8,2],gap='medium')
type_graph=st.sidebar.radio("Choisissez le type de graphique", ['empilé','étalé'])

with quant:
    fig_scatter_matrix = px.scatter(df, x=selected_variable_3, y=selected_variable_4,color_discrete_sequence=colors)
    fig_scatter_matrix.update_layout(title=f'Nuage de points entre {selected_variable_3} et {selected_variable_4}')
    fig_scatter_matrix.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.25)
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)
with qual:
    fig_croisé = px.bar(df, x=selected_variable_1, color=selected_variable_2,barmode=barmode_selected(type_graph),color_discrete_sequence= colors)
    fig_croisé.update_layout(title=f'Graphique en barres groupées - {selected_variable_1 } vs {selected_variable_2 }')
    fig_croisé.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.28)
    if selected_variable_1=="MOIS D'OCTROI DU CREDIT" or selected_variable_2=="MOIS D'OCTROI DU CREDIT":
        fig_croisé.update_xaxes(categoryorder='array', categoryarray=order_of_months)
    elif selected_variable_1=="ANNEE D'OCTROI DU CREDIT" or selected_variable_2=="ANNEE D'OCTROI DU CREDIT":
        fig_croisé.update_xaxes(categoryorder='array', categoryarray=order_of_years)     
    st.plotly_chart(fig_croisé)

occurences_an=df["ANNEE D'OCTROI DU CREDIT"].value_counts()
df["Nombre de prêts sur l'année"]=occurences_an[df["ANNEE D'OCTROI DU CREDIT"]].values
occurences_mo=df["MOIS D'OCTROI DU CREDIT"].value_counts()
df["Nombre de prêts sur le mois"]=occurences_mo[df["MOIS D'OCTROI DU CREDIT"]].values
fig_ann = px.area(df, x="ANNEE D'OCTROI DU CREDIT", y="Nombre de prêts sur l'année", color="MOIS D'OCTROI DU CREDIT",line_group="Nombre de prêts sur le mois",color_discrete_sequence= colors,custom_data=[df["MOIS D'OCTROI DU CREDIT"],df["ANNEE D'OCTROI DU CREDIT"],df["Nombre de prêts sur l'année"],df['Nombre de prêts sur le mois']])
fig_ann.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},height= 500,width= 1280)
fig_ann.update_xaxes(categoryorder='array', categoryarray=order_of_years)
fig_ann.update_traces(hovertemplate='<b>Année</b>: %{customdata[1]}<br>'
                                    "<b>Nbre de prêts sur l'année</b>: %{customdata[2]}<br>"
                                    "<b>Mois</b>: %{customdata[0]}<br>"
                                    '<b>Nbre de prêts sur le mois</b>: %{customdata[3]}<br>',
                                    hoverlabel=dict(font=dict(size=16, color='white'))),
#st.plotly_chart(fig_ann)

#Map_agencies
data = openpyxl.load_workbook("Classeur2.xlsx")
datas = data.active
donnees = []
for ligne in datas.iter_rows(values_only=True):
    donnees.append(list(ligne))
en_tetes = donnees[0]
donnees = donnees[1:]
df2 = pd.DataFrame(donnees, columns=en_tetes)
df= pd.merge(df, df2, on='AGENCES')

occurences=df['AGENCES'].value_counts()
df['size']=occurences[df['AGENCES']].values


df['Nbre de crédits risqués par agence']=d[df['AGENCES']].values
df["Taux de défauts de paiement"]=df['Nbre de crédits risqués par agence']/df['size']
df["Taux de défauts de paiement en pct"]=df["Taux de défauts de paiement"].map(lambda x: f'{x:.2%}')
center_lat = 5.31908
center_lon = -4.01299
px.set_mapbox_access_token("pk.eyJ1IjoicXM2MjcyNTI3IiwiYSI6ImNraGRuYTF1azAxZmIycWs0cDB1NmY1ZjYifQ.I1VJ3KjeM-S613FLv3mtkw")

fig_map = px.scatter_mapbox(df, lat=df.latitude, lon=df.longitude, color="Taux de défauts de paiement", size="size",
                  color_continuous_scale=px.colors.diverging.RdYlGn_r, zoom=12,custom_data=[df['AGENCES'],df['size'],df['Taux de défauts de paiement en pct']],size_max=40)

fig_map.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.05)','height': 700,'width': 1540,'mapbox': {'center': {'lat': center_lat, 'lon': center_lon}}})
fig_map.update_traces(text=df['Taux de défauts de paiement en pct'], hovertemplate='<b>Agence</b>: %{customdata[0]}<br>'
                                                                                      '<b>Latitude</b>: %{lat:.4f}<br>'
                                                                                      '<b>Longitude</b>: %{lon:.4f}<br>'
                                                                                      '<b>Nbre de prêts</b>: %{customdata[1]}<br>'
                                                                                      '<b>Taux de prêts en défaut de paiement</b>: %{customdata[2]}',
                                                                                      hoverlabel=dict(font=dict(size=16, color='white')),

                                                                                      )

st.plotly_chart(fig_map)
