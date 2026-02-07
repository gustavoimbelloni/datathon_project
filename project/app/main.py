from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import sys

# Adicionar o diretório src ao path para importar módulos de pré-processamento
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_data
from feature_engineering import apply_feature_engineering

app = Flask(__name__)

# Carregar o modelo e as features
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.joblib')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'features.joblib')

# Carregamento global do modelo
model = None
model_features = None

def load_model():
    global model, model_features
    if model is None:
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Converter para DataFrame (aceita um único objeto ou lista de objetos)
        if isinstance(data, dict):
            data = [data]
        df_input = pd.DataFrame(data)
        
        # Aplicar pré-processamento e engenharia de features
        df_processed = preprocess_data(df_input)
        df_engineered = apply_feature_engineering(df_processed)
        
        # Garantir que temos todas as colunas necessárias
        for col in model_features:
            if col not in df_engineered.columns:
                df_engineered[col] = 0
        
        # Reordenar colunas para bater com o modelo
        X = df_engineered[model_features]
        
        # Predição
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "risco_defasagem": bool(pred),
                "probabilidade": float(prob)
            })
            
        return jsonify({"predictions": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
