import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from io import StringIO

def get_client_list():
    response = requests.get('http://localhost:5000/get_client_list')
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération de la liste des clients.")
        return []
    
# Appeler l'endpoint pour obtenir le DataFrame brut(prétraiter)
def get_dataframe_from_api(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        df = pd.read_json(StringIO(response.text))
        return df
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération du DataFrame : {str(e)}")
        return pd.DataFrame()

# Utilisation de la fonction pour obtenir le DataFrame 
url = 'http://localhost:5000/full_dataframe'
df = get_dataframe_from_api(url)


# Appeler l'endpoint pour obtenir le DataFrame d'origine
def get_original_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        original_data = pd.read_json(StringIO(response.text))
        return original_data
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération du DataFrame : {str(e)}")
        return pd.DataFrame()

# Utilisation de la fonction pour obtenir le DataFrame
url = 'http://localhost:5000/original_data'
original_data= get_original_data(url)


# def predict_credit_score(user_features):
#     # Appeler l'endpoint pour obtenir les prédictions
#     response = requests.post('http://localhost:5000/predict_credit_score', json={"features": user_features})

#     if response.status_code == 200:
#         prediction = response.json()["prediction"]
#         st.write(f"Probabilité de faillite prédite : {prediction}")
#     else:
#         st.error("Erreur lors de la prédiction du score de crédit.")

def get_prediction(data):
    response = requests.post('http://localhost:5000/predict')
    return response.json(data)

   
# def apply_explainer_local(client_id):
#     response = requests.post('http://localhost:5000/apply_explainer_local', json={'client_id': client_id})

#     if response.status_code == 200 and response.content:
#         return response.json().get('explainer_result')
#     else:
#         st.error("Erreur lors de l'application de l'explainer local.")
#         return None
    
def create_gauge(bid_price, ask_price, current_price, spread):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        title={'text': "Score"},
        delta={'reference': ask_price, 'relative': False, 'increasing': {'color': "RebeccaPurple"}, 'decreasing': {'color': "RoyalBlue"}},
        value=current_price,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'shape': 'angular',
            'axis': {'range': [bid_price - spread, ask_price + spread]},
            'bar': {'color': "darkblue"},
            'bgcolor': 'yellow',
            'borderwidth': 2,
            'bordercolor': 'black',
            'steps': [
                {'range': [bid_price * 0.9, bid_price], 'color': 'green'},
                {'range': [ask_price, ask_price * 1.1], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': 'orange', 'width': 6},
                'thickness': 0.75,
                'value': current_price,
            }
        }
    ))

    return fig

def main():
    st.title("Dashboard de Scoring de Crédit - Prêt à Dépenser")
    st.subheader("Auteur : TIDIANE Barry")

    
    selected_client = st.selectbox("Sélectionnez un client :", options=df['SK_ID_CURR'])
    st.write(selected_client)
    
    # st.write(f"Probabilité de faillite prédite : {client_info.get('prediction', 'N/A')}")

    # Visualisation des informations descriptives relatives à un client
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Données de 10 échantillons")
        st.subheader(f"Informations descriptives du client identifié par {selected_client} :")
        # Afficher les informations spécifiques du client
        st.write(f"Indicateur de défaut de paiement: {original_data[original_data['SK_ID_CURR'] == selected_client]['TARGET'].iloc[0]}")
        st.write(f"Genre du client: {original_data[original_data['SK_ID_CURR'] == selected_client]['CODE_GENDER'].iloc[0]}")
        st.write(f"Âge : {int(original_data[original_data['SK_ID_CURR'] == selected_client]['DAYS_BIRTH'].iloc[0]/-365)}")
        st.write(f" Indicateur de possession d'une voiture: {original_data[original_data['SK_ID_CURR'] == selected_client]['FLAG_OWN_CAR'].iloc[0]}")
        st.write(f"Indicateur de possession de biens immobiliers: {df[df['SK_ID_CURR'] == selected_client]['FLAG_OWN_REALTY'].iloc[0]}")
        st.write(f"Montant mensuel à rembourer: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_ANNUITY'].iloc[0]}")
        st.write(f"Montant total du crédit demandé: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_CREDIT'].iloc[0]}")
        st.write(f"Montant total du revenu du client: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_INCOME_TOTAL'].iloc[0]}")
        # st.write(f"Revenu provenant du travail: {df[df['SK_ID_CURR'] == selected_client]['NAME_INCOME_TYPE_Working'].iloc[0]}")
        #st.write(df)


     # Ajouter la possibilité de saisir des features pour la prédiction
    st.sidebar.subheader("Saisir des features pour la prédiction :")
    feature1 = st.sidebar.number_input("CODE_GENDER", value=df[df['SK_ID_CURR'] == selected_client]['CODE_GENDER'].iloc[0])
    feature2 = st.sidebar.number_input("DAYS_BIRTH", value=int(df[df['SK_ID_CURR'] == selected_client]['DAYS_BIRTH'].iloc[0] / -365))
    feature3 = st.sidebar.number_input("AMT_ANNUITY", value=df[df['SK_ID_CURR'] == selected_client]['AMT_ANNUITY'].iloc[0])
    feature4 = st.sidebar.number_input("AMT_CREDIT", value=df[df['SK_ID_CURR'] == selected_client]['AMT_CREDIT'].iloc[0])

    user_features = [feature1, feature2, feature3, feature4]

    if st.sidebar.button("Prédire le score de crédit", key="predict_button", on_click=lambda: get_prediction(df[df['SK_ID_CURR'] == selected_client].to_json(orient = "split"))):
        # Utilisez la méthode POST en cliquant sur le bouton de prédiction

        pass



    # # Visualisation du score et interprétation
    # st.header("Score de Crédit et Interprétation")
    # st.write(f"Score du client : {client_info.get('prediction', 'N/A')}")

    # Afficher les informations spécifiques du client
    client_info = df[df['SK_ID_CURR'] == selected_client].to_dict(orient='records')[0]

    # Jauge de score
    # Créer la jauge
    bid_price = 100
    ask_price = 120
    current_price = float(client_info.get('prediction', 0.5))  # Remplacer par la valeur réelle du score
    spread = 5

    gauge_placeholder = st.empty()

    # Créer la jauge et afficher dans le placeholder
    fig = create_gauge(bid_price, ask_price, current_price, spread)
    with gauge_placeholder:
        st.plotly_chart(fig, use_container_width=True)


   
    # # Système de filtre
    # st.sidebar.header("Filtres")
    # # Ajoutez des options de filtre selon vos besoins

    # # Comparaison des informations descriptives
    # st.header("Comparer les informations descriptives")
    # # Ajoutez des éléments interactifs pour la comparaison

    # # Appliquer l'explainer local et afficher les résultats
    # explainer_result = apply_explainer_local(selected_client)

    # if explainer_result:
    #     st.header("Explainer Local")
    #     st.pyplot(explainer_result)  # Afficher le graphique de l'explainer local)

if __name__ == "__main__":
    main()
