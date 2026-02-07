import pandas as pd

def apply_feature_engineering(df):
    """Aplica engenharia de atributos aos dados pré-processados."""
    df_eng = df.copy()
    
    # Criar atributo: Tempo de casa (Anos desde o ingresso até 2024)
    if 'Ano ingresso' in df_eng.columns:
        df_eng['Tempo_Casa'] = 2024 - df_eng['Ano ingresso']
        # Remover Ano ingresso para evitar redundância
        df_eng = df_eng.drop(columns=['Ano ingresso'])
        
    # Criar atributo: Média de Notas (Matemática e Português)
    if 'Matem' in df_eng.columns and 'Portug' in df_eng.columns:
        df_eng['Media_Notas'] = (df_eng['Matem'] + df_eng['Portug']) / 2
        
    # Criar atributo: Soma de Indicadores (IAA, IEG, IPS, IDA, IPV, IAN)
    indicadores = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN']
    existing_indicadores = [col for col in indicadores if col in df_eng.columns]
    if existing_indicadores:
        df_eng['Soma_Indicadores'] = df_eng[existing_indicadores].sum(axis=1)
        
    return df_eng

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import load_data, preprocess_data
    data = load_data('C:/Users/gustavoborde/Downloads/BASEDEDADOSPEDE2024-DATATHON.xlsx')
    processed = preprocess_data(data)
    engineered = apply_feature_engineering(processed)
    print(f"Dados com engenharia de atributos: {engineered.shape}")
    print(engineered.head())
