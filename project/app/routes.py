from flask import request, jsonify
import pandas as pd
import os
import sys

# Adicionar src ao path para importar módulos de pré-processamento
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_data
from feature_engineering import apply_feature_engineering


def register_routes(app, load_model):
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "healthy"}), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            model, model_features = load_model()

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            # Aceita objeto único ou lista
            if isinstance(data, dict):
                data = [data]

            df_input = pd.DataFrame(data)

            # Pré-processamento + feature engineering
            df_processed = preprocess_data(df_input)
            df_engineered = apply_feature_engineering(df_processed)

            # Garante colunas esperadas pelo modelo
            for col in model_features:
                if col not in df_engineered.columns:
                    df_engineered[col] = 0

            X = df_engineered[model_features]

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            results = [
                {
                    "risco_defasagem": bool(pred),
                    "probabilidade": float(prob),
                }
                for pred, prob in zip(predictions, probabilities)
            ]

            return jsonify({"predictions": results}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500