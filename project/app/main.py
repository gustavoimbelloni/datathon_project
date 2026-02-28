from flask import Flask
import joblib
import os

from routes import register_routes

app = Flask(__name__)

# Caminhos de artefatos
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
    return model, model_features


# Registra rotas separadas
register_routes(app, load_model)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
