import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
import joblib
import shap
import numpy as np  
import base64

# Fonction pour obtenir la liste des clients
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

# Utilisation de la fonction pour obtenir le DataFrame d'origine
url = 'http://localhost:5000/original_data'
original_data= get_original_data(url)

# Fonction pour effectuer la prédiction
def get_prediction(data):
    try:
        response = requests.post('http://localhost:5000/predict', json=data)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        response_data = response.json()
        return response_data["prediction"][0]
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")
        return {"error": "Échec de récupération de la prédiction"}

# Fonction pour obtenir les valeurs SHAP
def get_shap_values(data):
    response = requests.post('http://localhost:5000/explain', json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        shap_values_list = response_data.get("shap_values")
        shap_summary_plot_data = response_data.get("shap_summary_plot_data")

        if shap_summary_plot_data:
            # Convertir la chaîne base64 en bytes
            shap_summary_plot_data_bytes = base64.b64decode(shap_summary_plot_data)
            
            # Afficher l'image dans Streamlit
            st.markdown(f'<img src="data:image/png;base64,{base64.b64encode(shap_summary_plot_data_bytes).decode()}" alt="SHAP Summary Plot">', unsafe_allow_html=True)
            
            # Retourner les valeurs SHAP pour une utilisation ultérieure si nécessaire
            return shap_values_list
        else:
            st.warning("Échec de la génération du tracé récapitulatif SHAP.")
            return None

    else:
        st.warning("Échec de l'obtention des valeurs SHAP.")
        return None
    

# Fonction pour obtenir la contribution SHAP d'une seule fonctionnalité du backend
def get_shap_contribution(data):
    response = requests.post('http://localhost:5000/get_shap_contribution', json=data)
    
    if response.status_code == 200:
        return True
    else:
        st.warning("Échec de l'obtention de la contribution SHAP.")
        return False


def columns_names():
        cols_names = joblib.load('cols_names.joblib')
        return cols_names



def create_gauge(bid_price, ask_price, current_price, spread):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        title={'text': "Probabilité de faillite prédite en pourcentage"},
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
    

    # Visualisation des informations descriptives relatives à un client
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Données de 10 échantillons")
        st.subheader(f"Informations descriptives du client identifié par {selected_client} :")
        # Afficher les informations spécifiques du client
        st.write(f"Indicateur de défaut de paiement: {original_data[original_data['SK_ID_CURR'] == selected_client]['TARGET'].iloc[0]}")
        st.write(f"Genre du client: {original_data[original_data['SK_ID_CURR'] == selected_client]['CODE_GENDER'].iloc[0]}")
        st.write(f"Âge : {int(original_data[original_data['SK_ID_CURR'] == selected_client]['DAYS_BIRTH'].iloc[0]/-365)}")
        st.write(f" Indicateur de possession d'une voiture: {original_data[original_data['SK_ID_CURR'] == selected_client]['FLAG_OWN_CAR'].iloc[0]}")
        st.write(f"Indicateur de possession de biens immobiliers: {original_data[original_data['SK_ID_CURR'] == selected_client]['FLAG_OWN_REALTY'].iloc[0]}")
        st.write(f"Montant mensuel à rembourer: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_ANNUITY'].iloc[0]}")
        st.write(f"Montant total du crédit demandé: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_CREDIT'].iloc[0]}")
        st.write(f"Montant total du revenu du client: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_INCOME_TOTAL'].iloc[0]}")
        # st.write(f"Revenu provenant du travail: {df[df['SK_ID_CURR'] == selected_client]['NAME_INCOME_TYPE_Working'].iloc[0]}")
        #st.write(df)

    # Saisie des features pour la prédiction
    st.sidebar.subheader("Saisir des features pour la prédiction :")
    feature1 = st.sidebar.number_input("CODE_GENDER", value=df[df['SK_ID_CURR'] == selected_client]['CODE_GENDER'].iloc[0])
    feature2 = st.sidebar.number_input("DAYS_BIRTH", value=int(df[df['SK_ID_CURR'] == selected_client]['DAYS_BIRTH'].iloc[0] / -365))
    feature3 = st.sidebar.number_input("AMT_ANNUITY", value=df[df['SK_ID_CURR'] == selected_client]['AMT_ANNUITY'].iloc[0])
    feature4 = st.sidebar.number_input("AMT_CREDIT", value=df[df['SK_ID_CURR'] == selected_client]['AMT_CREDIT'].iloc[0])

    user_features = [feature1, feature2, feature3, feature4]

    # Prédiction du score de crédit
    if st.sidebar.button("Prédire le score de crédit", key="predict_button"):
        #features = df.columns[3:588].values
        # Remplacer par les valeurs d'origines des features
        features = columns_names()
        selected_client_data = df[df['SK_ID_CURR'] == selected_client][features]
        prediction = get_prediction(selected_client_data.to_dict(orient='records')[0])
        # Arrondir la prédiction à deux chiffres après la virgule et convertir en pourcentage
        prediction = round(prediction, 2) * 100
        prediction_with_percent = f"{prediction}%"
        st.session_state.selected_client_data = selected_client_data  # Stocker les données dans la variable de session
        st.write(prediction_with_percent)

        fig = create_gauge(0, 100, prediction, 0.1)
        st.plotly_chart(fig, use_container_width=True)


    # Calcul de la contribution SHAP pour une seule feature
    if st.sidebar.button("Expliquer la prédiction", key="explain_button"):
        if st.session_state.selected_client_data is not None:
            shap_values = get_shap_values(st.session_state.selected_client_data.to_dict(orient='records')[0])

            # Affichez les valeurs SHAP pour le débogage
            st.write("SHAP Values:", shap_values)

            if shap_values is not None and len(shap_values) == 1:
                # Accéder à la première sous-liste des valeurs SHAP
                shap_values_list = shap_values[0]

                # Assurez-vous que la sous-liste a la longueur attendue
                if len(shap_values_list) == 2:
                    # Visualiser les valeurs SHAP sous forme de résumé
                    shap.summary_plot(shap_values_list, st.session_state.selected_client_data)
                else:
                    st.warning("Les valeurs SHAP ne sont pas dans le format attendu.")
            else:
                st.warning("Échec de l'obtention des valeurs SHAP..")

    
    # Sélectionner une observation spécifique à interpréter (par exemple, la première observation)
    selected_observation  = {
        "CODE_GENDER": "CODE_GENDER",
        "DAYS_BIRTH": "DAYS_BIRTH",  
        "AMT_ANNUITY": "AMT_ANNUITY",  
        "AMT_CREDIT": "AMT_CREDIT",  
        "EXT_SOURCE_2": "EXT_SOURCE_2"  
    }

    # Sélectionner une observation spécifique à interpréter
    selected_observation = st.sidebar.selectbox("Sélectionnez une feature pour la contribution SHAP", options=list(selected_observation.keys()))

    # Calcul de la contribution SHAP pour la feature sélectionnée
    if st.sidebar.button("Afficher la contribution SHAP", key="shap_button"):
        # Envoyer la feature sélectionnée au backend
        feature_name = selected_observation
        data = {"feature_name": feature_name}
        success = get_shap_contribution(data)
        
        if success:
            st.success(f"La contribution SHAP pour {feature_name} a été calculée avec succès et sauvegardée.")
           
            shap.dependence_plot(feature_name, shap_values, pd.DataFrame([selected_observation]), show=True)
        else:
            st.error("Erreur lors du calcul de la contribution SHAP.")


if __name__ == "__main__":
    main()
