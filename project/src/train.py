import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Adicionar o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess_data, split_data
from feature_engineering import apply_feature_engineering

def train_model(X_train, y_train):
    """Treina o modelo de Random Forest."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Salva o modelo treinado em um arquivo .joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

if __name__ == "__main__":
    # 1. Carregar e Processar
    data_path = 'C:/Users/gustavoborde/Downloads/BASEDEDADOSPEDE2024-DATATHON.xlsx'
    df = load_data(data_path)
    df_processed = preprocess_data(df)
    df_engineered = apply_feature_engineering(df_processed)
    
    # 2. Dividir
    X_train, X_test, y_train, y_test = split_data(df_engineered)
    
    # 3. Treinar
    print("Iniciando treinamento do modelo...")
    model = train_model(X_train, y_train)
    
    # 4. Avaliar
    y_pred = model.predict(X_test)
    print("Avaliação do Modelo:")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # 5. Salvar
    model_path = 'C:/Users/gustavoborde/Downloads/datathon_project/project/app/model/model.joblib'
    save_model(model, model_path)
    print(f"Modelo salvo em: {model_path}")
    
    # Salvar também as colunas usadas para garantir consistência na API
    joblib.dump(X_train.columns.tolist(), 'C:/Users/gustavoborde/Downloads/datathon_project/project/app/model/features.joblib')
