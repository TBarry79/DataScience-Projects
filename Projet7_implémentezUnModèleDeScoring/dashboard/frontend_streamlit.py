import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
import joblib
import shap
import numpy as np  
from streamlit_shap import st_shap
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


# Fonction pour récupérer les noms des features
def columns_names():
        cols_names = joblib.load('cols_names.joblib')
        return cols_names


# Fonction pour récupérer le modele du backend
def load_model_from_backend():
    url = "http://localhost:5000/load_model"  # Remplacez par l'URL de votre serveur Flask
    try:
        response = requests.get(url)
        response_data = response.json()
        
        if "model_path" in response_data:
            # Charger le modèle depuis le fichier temporaire
            return joblib.load(response_data["model_path"])
        elif "error" in response_data:
            st.error(f"Erreur lors du chargement du modèle: {response_data['error']}")
        else:
            st.error("Réponse inattendue du serveur.")
    except Exception as e:
        st.error(f"Erreur de récupération du modèle: {str(e)}")
    return None

# Charger le modèle depuis le backend
model = load_model_from_backend()


# Fonction pour la construction du jauge de score
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

    selected_client = st.selectbox("Sélectionnez un client :", options=df['ID'])    

    # Visualisation des informations descriptives relatives à un client
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Données de 10 échantillons")
        st.subheader(f"Informations descriptives du client identifié par {selected_client} :")
        # Afficher les informations spécifiques du client
        st.write(f"Genre du client: {original_data[original_data['SK_ID_CURR'] == selected_client]['CODE_GENDER'].iloc[0]}")
        st.write(f"Âge : {int(original_data[original_data['SK_ID_CURR'] == selected_client]['DAYS_BIRTH'].iloc[0]/-365)}")
        st.write(f" Indicateur de possession d'une voiture: {original_data[original_data['SK_ID_CURR'] == selected_client]['FLAG_OWN_CAR'].iloc[0]}")
        st.write(f"Indicateur de possession de biens immobiliers: {original_data[original_data['SK_ID_CURR'] == selected_client]['FLAG_OWN_REALTY'].iloc[0]}")
        st.write(f"Montant mensuel à rembourer: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_ANNUITY'].iloc[0]}")
        st.write(f"Montant total du crédit demandé: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_CREDIT'].iloc[0]}")
        st.write(f"Montant total du revenu du client: {original_data[original_data['SK_ID_CURR'] == selected_client]['AMT_INCOME_TOTAL'].iloc[0]}")
        #st.write(original_data)

     # Saisie des features pour la prédiction
    st.sidebar.subheader("Saisir des features pour la prédiction :")
    feature1 = st.sidebar.number_input("CODE_GENDER", value=df[df['ID'] == selected_client]['CODE_GENDER'].iloc[0])
    feature2 = st.sidebar.number_input("DAYS_BIRTH", value=int(df[df['ID'] == selected_client]['DAYS_BIRTH'].iloc[0] / -365))
    feature3 = st.sidebar.number_input("AMT_ANNUITY", value=df[df['ID'] == selected_client]['AMT_ANNUITY'].iloc[0])
    feature4 = st.sidebar.number_input("AMT_CREDIT", value=df[df['ID'] == selected_client]['AMT_CREDIT'].iloc[0])

    user_features = [feature1, feature2, feature3, feature4]

    # Prédiction du score de crédit
    if st.sidebar.button("Prédire le score de crédit", key="predict_button"):
        features = columns_names()
        selected_client_data = df[df['ID'] == selected_client][features]
        prediction = get_prediction(selected_client_data.to_dict(orient='records')[0])
        # Arrondir la prédiction à deux chiffres après la virgule et convertir en pourcentage
        if prediction is not None:
            prediction = round(prediction, 2) * 100
            st.write("Probabilité de défaut de crédit :", prediction, "%")
        else:
            st.warning("Aucune prédiction disponible.")
        st.session_state.selected_client_data = selected_client_data  # Stocker les données dans la variable de session

        fig = create_gauge(0, 100, prediction, 0.1)
        st.plotly_chart(fig, use_container_width=True)

    
    # # Bouton pour afficher le graphique pour le client selectionner avec un score prédit > 0,5
    # if st.sidebar.button("Afficher le du client à risque"):
    #     features = columns_names()
    #     selected_client_data = df[df['ID'] == selected_client][features]
    #     prediction = get_prediction(selected_client_data.to_dict(orient='records')[0])

    #      # S'assurer que la prédictions n'est pas nulle
    #     if prediction is not None:
    #         # Compter les occurrences de prédictions > 0,5 pour tous les clients
    #         count_high_risk_clients = np.sum(prediction > 0.5)

    #         # Afficher le nombre de clients à haut risque
    #         st.write("Nombre de clients avec un score > 0.5 :", count_high_risk_clients)

    #         # Créer un graphique à barres pour visualiser la distribution des scores prédits
    #         fig, ax = plt.subplots()
    #         ax.hist(prediction, bins=20, color='blue', alpha=0.7, label='Bon clients')

    #         # Vérifier s'il y a des clients à risque avant la représentation
    #         if count_high_risk_clients > 0:
    #             ax.hist([prediction] * count_high_risk_clients, bins=20, color='red', alpha=0.7, label='Client à risque')
    #         ax.set_xlabel('Score de crédit')
    #         ax.set_ylabel('Nombre de clients')
    #         ax.legend()

    #     else:
    #         st.warning("Aucune prédiction disponible.")

    #     # Afficher le graphique
    #     st.pyplot(fig)


    # Bouton pour afficher le graphique pour les clients avec un score prédit > 0,5
    if st.sidebar.button("Afficher le nombre de clients à risque"):
        features = columns_names()

        # Obtenir les prédictions pour tous les clients
        predictions = df.apply(lambda row: get_prediction(row[features].to_dict()), axis=1)

        # S'assurer que les prédictions ne sont pas nulles
        if not predictions.isnull().all():
            # Compter les occurrences de prédictions > 0,5 pour tous les clients
            count_high_risk_clients = np.sum(predictions > 0.5)

            # Afficher le nombre de clients à haut risque
            st.write("Nombre de clients avec un score > 0.5 :", count_high_risk_clients)

            # Créer un graphique à barres pour visualiser la distribution des scores prédits
            fig, ax = plt.subplots()
            ax.hist(predictions, bins=20, color='blue', alpha=0.7, label='Bons clients')

            # Vérifier s'il y a des clients à risque avant la représentation
            if count_high_risk_clients > 0:
                ax.hist(predictions[predictions > 0.5], bins=20, color='red', alpha=0.7, label='Clients à risque')
            ax.set_xlabel('Score de crédit')
            ax.set_ylabel('Nombre de clients')
            ax.legend()

        else:
            st.warning("Aucune prédiction disponible.")

        # Afficher le graphique
        st.pyplot(fig)


    # Bouton pour expliquer la prédiction
    if st.sidebar.button("Expliquer la prédiction", key="explain_button"):
        # Charger le modèle depuis le backend
        model = load_model_from_backend()

        # Supprimer la colonne 'ID' de df
        df_without_id = df.drop('ID', axis=1)

        # Créer un explainer SHAP avec le modèle et le DataFrame sans la colonne 'ID'
        explainer = shap.Explainer(model, df_without_id)

        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(df_without_id)
        
        # Créer un objet Explanation
        explanation = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=df_without_id)
        
        # Affichage graphiques des valeurs shap
        st_shap(shap.plots.waterfall(explanation[0]), height=300)
        st_shap(shap.plots.beeswarm(explanation), height=300)


        # Sélectionner la variable pour laquelle vous souhaitez créer le graphique de dépendance
        feature_name = "EXT_SOURCE_3"  # Remplacez par le nom de la variable d'intérêt

        # Créer le graphique de dépendance SHAP
        st_shap(shap.dependence_plot(feature_name, shap_values, df_without_id, show=False))



    user_features = {}
    selected_observation = {
        "CODE_GENDER": "CODE_GENDER",
        "DAYS_BIRTH": "DAYS_BIRTH",
        "AMT_ANNUITY": "AMT_ANNUITY",
        "AMT_CREDIT": "AMT_CREDIT",
        "EXT_SOURCE_3": "EXT_SOURCE_3"
    }

    
    # Selectionner une fonctionnalité
    selected_feature = st.sidebar.selectbox("Sélectionnez une feature pour la contribution SHAP", options=list(selected_observation.keys()))

    # Define default values based on descriptive statistics (replace with actual values)
    default_feature_values = {
        "CODE_GENDER": df[selected_feature].mode().iloc[0],
        "DAYS_BIRTH": df[selected_feature].median(),
        "AMT_ANNUITY": df[selected_feature].mean(),
        "AMT_CREDIT": df[selected_feature].mean(),
        "EXT_SOURCE_3": df[selected_feature].median()
    }

    # Obtenir la valeur réelle de la fonctionnalité auprès de l'utilisateur
    actual_feature_value = st.sidebar.number_input(f"Valeur pour {selected_feature}", value=default_feature_values.get(selected_feature, 0.0))

    # Bouton pour déclencher le tracé SHAP
    if st.sidebar.button("Afficher la contribution SHAP", key="shap_button"):
        # Mettre à jour user_features avec la valeur réelle de la fonctionnalité
        user_features[selected_feature] = actual_feature_value

        # Supprimer la colonne 'ID' de df
        df_without_id = df.drop('ID', axis=1)
    
        # Créer un DataFrame avec les fonctionnalités sélectionnées et la valeur utilisateur mise à jour
        user_df = pd.DataFrame([user_features], columns=df_without_id.columns)

        # Charger le modèle depuis le backend
        model = load_model_from_backend()

        # Calculer les valeurs SHAP pour l'observation mise à jour
        explainer = shap.Explainer(model, df_without_id)
        shap_values = explainer.shap_values(user_df)

        # Afficher le tracé de dépendance SHAP
        st_shap(shap.dependence_plot(str(selected_feature), shap_values, user_df, show=True))

if __name__ == "__main__":
    main()
