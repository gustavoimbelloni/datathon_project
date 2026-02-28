from pathlib import Path
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

# Adicionar o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess_data, split_data
from feature_engineering import apply_feature_engineering
from evaluate import evaluate_model


def train_model(X_train, y_train):
    """Treina o modelo Random Forest."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, file_path):
    """Salva o modelo treinado em arquivo .joblib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)


def run_training_pipeline(data_path: str | None = None):
    """Executa pipeline completo de treino e salva artefatos."""
    base_dir = Path(__file__).resolve().parents[1]  # project/
    default_data_path = base_dir / "data" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

    # Prioridade: argumento > variável de ambiente > caminho padrão do projeto
    final_data_path = Path(data_path) if data_path else Path(os.getenv("DATA_PATH", default_data_path))

    if not final_data_path.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {final_data_path}\n"
            f"Defina DATA_PATH ou coloque o arquivo em: {default_data_path}"
        )

    print("📂 Carregando dados...")
    df = load_data(str(final_data_path))
    df_processed = preprocess_data(df)
    df_engineered = apply_feature_engineering(df_processed)

    print("✂️  Dividindo dados em treino/teste...")
    X_train, X_test, y_train, y_test = split_data(df_engineered)
    print(f"   Treino: {len(X_train)} | Teste: {len(X_test)}")

    print("🧠 Treinando modelo...")
    model = train_model(X_train, y_train)

    print("📊 Avaliando modelo...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    evaluator = evaluate_model(y_test, y_pred, y_pred_proba=y_pred_proba, plot_all=False)

    model_dir = base_dir / "app" / "model"
    save_model(model, model_dir / "model.joblib")
    save_model(list(X_train.columns), model_dir / "features.joblib")

    print(f"✅ Modelo salvo em: {model_dir / 'model.joblib'}")
    print(f"✅ Features salvas em: {model_dir / 'features.joblib'}")

    return model, evaluator


if __name__ == "__main__":
    run_training_pipeline()
