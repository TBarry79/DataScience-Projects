from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
#from sklearn.inspection import plot_local_interpretation

app = Flask(__name__)

# Charger le modèle sauvegardé 
model = joblib.load('best_logistic_regression_model.joblib')

# Charger les données des clients 
df = joblib.load('sample_data_10.joblib')
# Exposer l'endpoint pour obtenir le DataFrame complet
@app.route('/full_dataframe', methods=['GET'])
def full_dataframe():
    try:
        # Charger le DataFrame depuis le fichier joblib 
        df = joblib.load('sample_data_10.joblib')

        # Retourner le DataFrame brut au format JSON
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
    # Récupérer la liste des clients à partir de la colonne 'SK_ID_CURR'
    client_list = df['SK_ID_CURR'].tolist()
    return jsonify(client_list)


# Exposer l'endpoint pour obtenir les informations d'un client spécifique
@app.route('/get_client_info', methods=['GET'])
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


# Ajouter cette route à votre application Flask
@app.route('/predict', methods=['POST'])
def predict_credit_score():
    data = request.get_json()
    prediction = model.predict(data)
    
    return jsonify({"prediction" : prediction.tolist()})



# Exposer l'endpoint pour appliquer l'explainer local
# @app.route('/apply_explainer_local', methods=['POST'])
# def apply_explainer_local():
#     data = request.get_json(force=True)
#     client_id = data.get('client_id')

#     # Récupérer les informations du client à partir de l'ID
#     client_info = df[df['SK_ID_CURR'] == client_id].to_dict(orient='records')[0]
    
#     # Ajouter la prédiction du modèle (remplacez cela par la logique réelle)
#     features = client_info.get('features', [])
#     prediction = model.predict_proba(np.array(features).reshape(1, -1))[0, 1]
#     client_info['prediction'] = prediction

#     # Appliquer l'explainer local (exemple fictif, remplacez-le par votre propre logique)
#     explainer_result = plot_local_interpretation(model, np.array(features).reshape(1, -1), features_names=df.columns)

#     return jsonify({'explainer_result': explainer_result})

if __name__ == '__main__':
    app.run(port=5000)
