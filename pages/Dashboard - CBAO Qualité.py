import streamlit as st
import pandas as pd
import plotly.express as px
import openpyxl
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards # beautify metric card with css
# Chargement des données
data = openpyxl.load_workbook(r"C:\Users\DELL I5\Desktop\CBAO\BBAID CBAO.xlsx")
datas = data.active
donnees = []
for ligne in datas.iter_rows(values_only=True):
    donnees.append(list(ligne))
en_tetes = donnees[0]
donnees = donnees[1:]
df = pd.DataFrame(donnees, columns=en_tetes)

pec_agence=df.groupby('Agence')['Note de la prise en charge'].mean()
# Thème des graphiques
df["Mois"] = df["Horodateur"].dt.month
df["Jour"] = df["Horodateur"].dt.day_of_week
df["heure"]=df["Horodateur"].dt.hour

df["Mois"]=df["Mois"].map({1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"})
order_of_months = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
df["Mois*"] = pd.Categorical(df["Mois"], categories=order_of_months, ordered=True)

df["Jour"]=df["Jour"].map({0:"Lundi",1:"Mardi",2:"Mercredi",3:"Jeudi",4:"Vendredi",5:"Samedi",6:"Dimanche"})
order_of_days = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi']
df["Jour*"] = pd.Categorical(df["Jour"], categories=order_of_days, ordered=True)

st.set_page_config(layout="wide")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url(https://www.cbaobank.com/sites/default/files/inline-images/OUVERTURE%20AGENCE%20KM_1.jpg);
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

st.markdown('<div style="text-align:center;width:100%;"><h1 style="color:black;background-color:#f7a900;border:#fc1c24;border-style:solid;border-radius:5px;">TABLEAU DE BORD INTERACTIF DE LA BOITE A IDEES DIGITALE</h1></div>', unsafe_allow_html=True)

colors = px.colors.sequential.Rainbow_r
# Section des graphiques sommaires


st.write("\n")
st.write("\n")

st.markdown(page_bg_img, unsafe_allow_html=True)


with st.container():    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total des questionnaires reçus", df.shape[0],"0")
    col2.metric("Note moyenne de l'accueil", np.round(df["Note de l'accueil"].mean(),2), np.round(df["Note de l'accueil"].mean()-4,2))
    col3.metric("Note moyenne de la prise en charge", np.round(df["Note de la prise en charge"].mean(),2), np.round(df["Note de la prise en charge"].mean()-4,2))
    style_metric_cards(background_color='#0c0c0c',border_left_color="#f7a900",box_shadow=True)

def palmarès(month_sel):
    result_mois = df[df['Mois'] == month_sel].groupby(['Zone', 'Mois'])['Mois'].count().reset_index(name="Mensuel")
    result_zone = df.groupby(['Zone'])['Mois'].count().reset_index(name="Annuel")
    # Fusion des résultats dans un DataFrame
    return pd.merge(result_mois, result_zone, on='Zone', how='left')

def format_ranking_index(df, index_col='Position'):
    df[index_col] = df.index + 1
    df[index_col] = df[index_col].apply(lambda x: f"{x}ème" if x > 1 else f"{x}er")
    return df.set_index(index_col)



selected_month = st.selectbox("Sélectionnez le mois",df["Mois"].unique())
actu, palm,top=st.columns(3)

with actu:
    st.subheader(f"Chiffres du mois actuel ({selected_month})",divider='rainbow')
    st.metric("Nbre total de questionnaires",df[df['Mois']==selected_month].shape[0],"12")
    avis=(df[df['Mois']==selected_month]['Motifs de la note de l\'accueil'].count()+df[df['Mois']==selected_month]['Motifs de la note de la prise en charge'].count())/2 
    suggestions = df[df['Mois']==selected_month]['Suggestions'].count()
    st.metric("Nbre total d'avis",avis,"13")
    st.metric("Nbre total de suggestion", suggestions,"-12")

with palm:
    st.subheader("Palmarès Zone",divider="rainbow")
    pa=palmarès(selected_month)
    st.dataframe(format_ranking_index(pa))
with top:
    st.subheader("Top 5 Agences",divider='rainbow')
    tops=df[df['Mois']==selected_month].groupby('Agence')['Agence'].value_counts().reset_index(name="Nbre de quest. remplis")
    tops.sort_values(by=["Nbre de quest. remplis"],inplace=True,ascending=False)
    tops.reset_index(inplace=True, drop=True)
    tops=format_ranking_index(tops)
    st.dataframe(tops)

# Histogramme et Camembert sur la même ligne
cam, hist = st.columns(2,gap='medium')

with cam:
    st.sidebar.subheader("CAMEMBERT")
    selected_categorical_variable_p = st.sidebar.selectbox("***:gray[Sélectionnez une variable catégorielle pour le camembert:]***", ['Agence', 'Niveau de réception',"Note de l'accueil","Note de la prise en charge",'Jour','Mois','Zone'])
    category_counts = df[selected_categorical_variable_p].value_counts()
    fig_pie = px.pie(names=category_counts.index, values=category_counts.values, title=f"Répartition de la variable {selected_categorical_variable_p}",color_discrete_sequence=colors)
    fig_pie.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.25)
    st.plotly_chart(fig_pie, use_container_width=True)
with hist:
    st.sidebar.subheader("HISTOGRAMME")
    selected_categorical_variable = st.sidebar.selectbox("***:gray[Sélectionnez la variable catégorielle pour l'histogramme :]***", ['Agence', 'Niveau de réception',"Note de l'accueil", "Note de la prise en charge",'Jour','Mois','Zone'])
    fig_histogram = px.histogram(df, x=df[selected_categorical_variable], color=df[selected_categorical_variable],title=f"Histogramme de {selected_categorical_variable}",color_discrete_sequence=colors)
    fig_histogram.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.35)
    fig_histogram.update_traces( textfont_color='rgba(255, 255, 255, 1)')
    if selected_categorical_variable=="Mois":
        fig_histogram.update_xaxes(categoryorder='array', categoryarray=order_of_months)
    elif selected_categorical_variable=="Jour":
        fig_histogram.update_xaxes(categoryorder='array', categoryarray=order_of_days)
    st.plotly_chart(fig_histogram,use_container_width=True)



# Section des analyses croisées
st.sidebar.subheader("ANALYSE CROISEE ENTRE VARIABLES NUMERIQUES")
int_columns = df.select_dtypes(include="int").columns
float_columns = df.select_dtypes(include="float").columns
selected_variable_3 = st.sidebar.selectbox("***:gray[Variable 1 :]***", int_columns.union(float_columns))
selected_variable_4 = st.sidebar.selectbox("***:gray[Variable 2 :]***",int_columns.union(float_columns))


st.sidebar.subheader("ANALYSE CROISEE ENTRE VARIABLES CATEGORIELLES")
selected_variable_1 = st.sidebar.selectbox("***:gray[Variable 1 :]***", ['Agence', 'Niveau de réception','Jour','Mois','Zone'])
selected_variable_2 = st.sidebar.selectbox("***:gray[Variable 2 :]***", ['Agence', 'Niveau de réception',"Note de la prise en charge","Note de l'accueil",'Jour','Mois','Zone'])

def barmode_selected(t):
    if t =='empilé':
        a='relative'  
    else: 
        a='group'
    return a

quant,qual=st.columns(2,gap='medium')
type_graph=st.sidebar.radio("Choisissez le type de graphique", ['empilé','étalé'])

with quant:
    fig_scatter_matrix = px.scatter(df, x=selected_variable_3, y=selected_variable_4,color_discrete_sequence=colors)
    fig_scatter_matrix.update_layout(title=f'Nuage de points entre {selected_variable_3} et {selected_variable_4}')
    fig_scatter_matrix.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.25)
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)
with qual:
    if selected_variable_2 in ["Note de l'accueil","Note de la prise en charge"]:
        fig_croisé = px.bar(df.groupby(selected_variable_1)[selected_variable_2].mean().reset_index(), x=selected_variable_1,y=selected_variable_2, color=selected_variable_2,barmode=barmode_selected(type_graph),color_discrete_sequence= colors)
    else:
        fig_croisé = px.bar(df, x=selected_variable_1, color=selected_variable_2,barmode=barmode_selected(type_graph),color_discrete_sequence= colors)
        
        if selected_variable_1=="Mois" or selected_variable_2=="Mois":
            fig_croisé.update_xaxes(categoryorder='array', categoryarray=order_of_months)
        elif selected_variable_1=="Jour" or selected_variable_2=="Jour":
            fig_croisé.update_xaxes(categoryorder='array', categoryarray=order_of_days)
    fig_croisé.update_layout(title=f'Graphique en barres groupées - {selected_variable_1 } vs {selected_variable_2 }')
    fig_croisé.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},title_x=0.28)

    st.plotly_chart(fig_croisé,use_container_width=True)


occurences_day=df["Jour"].value_counts()
df["Nombre de questionnaires remplis par jour"]=occurences_day[df["Jour"]].values
occurences_mo=df["Mois"].value_counts()
df["Nombre de questionnaires remplis dans le mois"]=occurences_mo[df["Mois"]].values
fig_ann = px.area(df, x="Mois*", y="Nombre de questionnaires remplis dans le mois", color="Jour*",line_group="Nombre de questionnaires remplis par jour",color_discrete_sequence= colors,custom_data=[df["Mois"],df["Jour"],df['Nombre de questionnaires remplis par jour'],df['Nombre de questionnaires remplis dans le mois']])
fig_ann.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0.3)',},height= 500,width= 1225)
fig_ann.update_xaxes(categoryorder='array', categoryarray=order_of_months)
fig_ann.update_traces(hovertemplate="<b>Mois</b>: %{customdata[0]}<br>"
                                    '<b>Jour</b>: %{customdata[1]}<br>'
                                    '<b>Nbre de quest. enregistrés sur le mois</b>: %{customdata[3]}<br>'
                                    "<b>Nbre de quest. enregistrés ce jour</b>: %{customdata[2]}<br>",
                                    hoverlabel=dict(font=dict(size=16, color='white'))),
