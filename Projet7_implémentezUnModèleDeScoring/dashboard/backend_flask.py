import matplotlib
matplotlib.use('Agg')

import io
import json
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import pandas as pd
import joblib
import numpy as np
import shap
import base64
#from sklearn.inspection import plot_local_interpretation

app = Flask(__name__)

# Charger le modèle sauvegardé 
model = joblib.load('best_logistic_regression_model.joblib')


# Charger les données des clients 
df = joblib.load('X_test_scaled.joblib')
# Exposer l'endpoint pour obtenir le DataFrame complet
@app.route('/full_dataframe', methods=['GET'])
def full_dataframe():
    try:
        # Charger le DataFrame depuis le fichier Joblib
        df = joblib.load('X_test_scaled.joblib')
        
        # Prendre un échantillon de 10 clients du DataFrame
        sample_df = df.sample(n=10, random_state=42)

        # Retourner l'échantillon au format JSON
        return jsonify(sample_df.to_dict(orient='records'))
    except Exception as e:
        # Gérer les erreurs, par exemple, en renvoyant un message d'erreur
        return jsonify({"error": str(e)})
    

# Charger les données des clients avant le prétraitement  
original_data = joblib.load('original_data.joblib')
# Ajouter cette route à votre application Flask
@app.route('/original_data', methods=['GET'])
def original_data():
    try:
        original_data = joblib.load('original_data.joblib')

        return jsonify(original_data.to_dict(orient='records'))
    except Exception as e:
        # Gérer les erreurs, par exemple, en renvoyant un message d'erreur
        return jsonify({"error": str(e)})  


# Exposer l'endpoint pour obtenir la liste des clients
@app.route('/get_client_list', methods=['GET'])
def get_client_list():
    #Prendre un échantillon de 10 clients du DataFrame
    sample_df = df.sample(n=10, random_state=42)

    # Récupérer la liste des 10 clients à partir de la colonne 'SK_ID_CURR'
    client_list = sample_df['SK_ID_CURR'].tolist()
    return jsonify(client_list)


# Exposer l'endpoint pour obtenir les informations d'un client spécifique
@app.route('/get_client_info', methods=['GET'])
def get_client_info():
    # data = request.get_json(force=False)
    # print("Received data:", data)
    # client_id = data.get('client_id')
    # print("client_id:", client_id)
    client_id = 100004
    
    if client_id is None:
        return jsonify({"error": "Client ID is missing"}), 400
    
    client_info = df[df['SK_ID_CURR'] == client_id]
    if client_info.empty:
        return jsonify({"error": "Client not found"}), 404
    
    client_info_dict = client_info.to_dict(orient='records')[0]

    return jsonify(client_info_dict)


# Ajouter une nouvelle route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict_credit_score():
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json()

        # Créer un DataFrame à partir des données JSON
        data_df = pd.DataFrame([data])

        # Effectuer la prédiction de probabilité
        prediction = model.predict_proba(data_df)[:, 1]

        # Retourner la prédiction au format JSON
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur
        return jsonify({"error": str(e)})


# Définir une fonction de sérialisation personnalisée
def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")    

#Exposer l'endpoint pour appliquer l'explainer local
@app.route('/explain', methods=['POST'])
def explain_prediction():
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json()

        # Créer un DataFrame à partir des données JSON
        data_df = pd.DataFrame([data])

        # Utiliser shap pour expliquer la prédiction
        explainer = shap.Explainer(model, data_df)
        shap_values = explainer.shap_values(data_df)

        # Afficher les valeurs Shap pour chaque caractéristique dans les journaux
        print("SHAP Values:", shap_values)

        # Visualisation SHAP summary plot
        plt.figure(figsize=(12, 6))  # Ajustez la taille de la figure selon vos besoins
        shap.summary_plot(shap_values, data_df, show=False)
        plt.tight_layout()


        # Convertir les valeurs Shap en liste
        shap_values_list = [value.tolist() if isinstance(value, np.ndarray) else value for value in shap_values]

        # Convertir l'image en une chaîne base64
        shap_summary_plot_bytes = io.BytesIO()
        plt.savefig(shap_summary_plot_bytes, format='png', bbox_inches='tight')
        shap_summary_plot_bytes.seek(0)
        shap_summary_plot_data = base64.b64encode(shap_summary_plot_bytes.read()).decode('utf-8')

        # Retourner les valeurs Shap et les données de l'image
        return json.dumps({"shap_values": shap_values_list, "shap_summary_plot_data": shap_summary_plot_data}, default=serialize)

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur avec les détails dans les journaux
        print("Error:", str(e))
        return jsonify({"error": str(e)})
    
# Prendre un échantillon de 10 clients du DataFrame
sample_df = df.sample(n=10, random_state=42)    
explainer = shap.Explainer(model, sample_df)

@app.route('/get_shap_contribution', methods=['POST'])
def get_shap_contribution():
    # Obtenir les données de la requête POST
    data = request.json

    app.logger.info(f"Received data: {data}")
    
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid data format. Expected a dictionary."}), 400

    observation = pd.DataFrame.from_dict(data, orient='columns')

    # Calculer les valeurs SHAP pour l'observation
    shap_values = explainer.shap_values(observation)

    # Sélectionner la variable pour laquelle vous souhaitez créer le graphique de dépendance SHAP
    feature_name = list(observation.columns)[0]

    # Créer le graphique de dépendance SHAP
    shap.dependence_plot(feature_name, shap_values, observation, show=False)

    # Sauvegarder le graphique au format image
    plt.savefig("shap_dependence_plot.png")

    return jsonify({"message": "SHAP contribution calculated and saved."})


if __name__ == '__main__':
    app.run(port=5000)
