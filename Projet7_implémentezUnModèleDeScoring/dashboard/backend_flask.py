import matplotlib
from sklearn.neighbors import NearestNeighbors
matplotlib.use('Agg')

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

# Enregistrer le modèle dans un fichier temporaire
joblib.dump(model, 'temp_model.joblib')

# Endpoint pour charger le modèle
@app.route('/load_model', methods=['GET'])
def load_model():
    try:
        # Return the path to the temporary model file
        return jsonify({"model_path": "temp_model.joblib"})
    except Exception as e:
        return jsonify({"error": str(e)})


# Charger les données des clients 
df = joblib.load('X_test_scaled.joblib')
# Exposer l'endpoint pour obtenir le DataFrame complet
@app.route('/full_dataframe', methods=['GET'])
def full_dataframe():
    try:
        # Charger le DataFrame depuis le fichier Joblib
        df = joblib.load('X_test_scaled.joblib')
        

        # Retourner l'échantillon au format JSON
        return jsonify(df.to_dict(orient='records'))
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

    # Récupérer la liste des 10 clients à partir de la colonne 'SK_ID_CURR'
    client_list = df['ID'].tolist()
    return jsonify(client_list)


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



if __name__ == '__main__':
     # Utilisation de Gunicorn au lieu du serveur de développement Flask
    app.run(host='0.0.0.0', port=5000)  
