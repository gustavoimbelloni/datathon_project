import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Carrega os dados do arquivo Excel."""
    return pd.read_excel(file_path)

def preprocess_data(df):
    """Realiza o pré-processamento básico dos dados."""
    # Selecionar colunas relevantes
    cols_to_keep = [
        'Fase', 'Idade 22', 'Gênero', 'Ano ingresso', 'IAA', 'IEG', 'IPS', 
        'IDA', 'Matem', 'Portug', 'IPV', 'IAN', 'Defas'
    ]
    # Filtrar apenas as colunas que existem no dataframe
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    df = df[existing_cols].copy()
    
    # Tratar valores nulos em notas (preencher com a média)
    if 'Matem' in df.columns:
        df['Matem'] = df['Matem'].fillna(df['Matem'].mean())
    if 'Portug' in df.columns:
        df['Portug'] = df['Portug'].fillna(df['Portug'].mean())
    
    # Codificar Gênero (M=1, F=0)
    if 'Gênero' in df.columns:
        df['Gênero'] = df['Gênero'].map({'M': 1, 'F': 0}).fillna(0)
    
    # Definir a variável alvo: Risco de Defasagem (1 se Defas < 0, 0 caso contrário)
    if 'Defas' in df.columns:
        df['target'] = (df['Defas'] < 0).astype(int)
        # Remover a coluna original Defas para evitar data leakage
        df = df.drop(columns=['Defas'])
    
    return df

def split_data(df, target_col='target', test_size=0.2, random_state=42):
    """Divide os dados em treino e teste."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Teste rápido
    data = load_data('data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
    processed_data = preprocess_data(data)
    print(f"Dados processados: {processed_data.shape}")
    print(processed_data.head())
