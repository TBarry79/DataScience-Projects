import json
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import shap
#from sklearn.inspection import plot_local_interpretation

app = Flask(__name__)

# Charger le modèle sauvegardé 
model = joblib.load('best_logistic_regression_model.joblib')

# Charger les données des clients 
df = joblib.load('test_set.joblib')
# Exposer l'endpoint pour obtenir le DataFrame complet
@app.route('/full_dataframe', methods=['GET'])
def full_dataframe():
    try:
        # Charger le DataFrame depuis le fichier Joblib
        df = joblib.load('test_set.joblib')

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
    # Récupérer la liste des clients à partir de la colonne 'SK_ID_CURR'
    client_list = df['SK_ID_CURR'].tolist()
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


    

#Exposer l'endpoint pour appliquer l'explainer local
@app.route('/explain', methods=['POST'])
def explain_prediction():
    try:
        # Récupérer les données JSON de la requête
        data = request.get_json()

        # Créer un DataFrame à partir des données JSON
        data_df = pd.DataFrame([data])

        # Utiliser shap pour expliquer la prédiction
        explainer = shap.LinearExplainer(model, data_df)
        shap_values = explainer.shap_values(data_df)

        # Afficher les valeurs Shap pour chaque caractéristique dans les journaux
        print("SHAP Values:", shap_values)

        # Retourner les valeurs Shap pour chaque caractéristique
        return jsonify({"shap_values": shap_values.tolist()})

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur avec les détails dans les journaux
        print("Error:", str(e))
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(port=5000)
