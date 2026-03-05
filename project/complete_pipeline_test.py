"""
Script de Validação Completa do Pipeline - Teste End-to-End

Este script testa todas as etapas do pipeline de ML:
1. Carregamento de dados (teste com dados sintéticos)
2. Preprocessamento
3. Feature Engineering
4. Split treino/teste
5. Treinamento
6. Predição
7. Avaliação
8. Persistência

Resultado esperado: TODO OK ✅
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_data, split_data
from feature_engineering import apply_feature_engineering
from train import train_model, save_model
from evaluate import ModelEvaluator, evaluate_model
import joblib

print("=" * 70)
print("🚀 TESTE COMPLETO DO PIPELINE DO PROJETO")
print("=" * 70)

# ============================================================================
# ETAPA 1: Preparar Dados Sintéticos
# ============================================================================
print("\n📂 ETAPA 1: Preparando dados sintéticos...")
try:
    np.random.seed(42)
    n_samples = 300
    
    sample_data = pd.DataFrame({
        'Fase': np.random.randint(1, 4, n_samples),
        'Idade 22': np.random.randint(8, 18, n_samples),
        'Gênero': np.random.choice(['M', 'F'], n_samples),
        'Ano ingresso': np.random.randint(2018, 2023, n_samples),
        'IAA': np.random.uniform(5, 10, n_samples),
        'IEG': np.random.uniform(5, 10, n_samples),
        'IPS': np.random.uniform(5, 10, n_samples),
        'IDA': np.random.uniform(5, 10, n_samples),
        'Matem': np.random.uniform(4, 10, n_samples),
        'Portug': np.random.uniform(4, 10, n_samples),
        'IPV': np.random.uniform(5, 10, n_samples),
        'IAN': np.random.uniform(5, 10, n_samples),
        'Defas': np.random.choice([-2, -1, 0, 1, 2], n_samples)
    })
    
    print(f"   ✅ {n_samples} amostras criadas")
    print(f"   ✅ {sample_data.shape[1]} features")
    print(f"   ✅ Primeiras linhas:\n{sample_data.head()}")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 2: Preprocessamento
# ============================================================================
print("\n🔄 ETAPA 2: Preprocessamento...")
try:
    df_processed = preprocess_data(sample_data)
    print(f"   ✅ Dados preprocessados")
    print(f"   ✅ Shape: {df_processed.shape}")
    print(f"   ✅ Colunas: {list(df_processed.columns)}")
    print(f"   ✅ Target distribuição:")
    print(f"      - Classe 0: {(df_processed['target'] == 0).sum()}")
    print(f"      - Classe 1: {(df_processed['target'] == 1).sum()}")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 3: Feature Engineering
# ============================================================================
print("\n⚙️  ETAPA 3: Feature Engineering...")
try:
    df_engineered = apply_feature_engineering(df_processed)
    print(f"   ✅ Features criadas com sucesso")
    print(f"   ✅ Shape: {df_engineered.shape}")
    print(f"   ✅ Features adicionadas:")
    added_features = set(df_engineered.columns) - set(df_processed.columns)
    for feat in added_features:
        print(f"      - {feat}")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 4: Split Treino/Teste
# ============================================================================
print("\n✂️  ETAPA 4: Dividindo dados em treino/teste...")
try:
    X_train, X_test, y_train, y_test = split_data(df_engineered)
    print(f"   ✅ Split realizado")
    print(f"   ✅ Treino: {len(X_train)} amostras")
    print(f"   ✅ Teste: {len(X_test)} amostras")
    print(f"   ✅ Proporção: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% treino")
    print(f"   ✅ Features por amostra: {X_train.shape[1]}")
    
    # Verificar separação
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    if len(train_indices.intersection(test_indices)) > 0:
        print(f"   ⚠️  AVISO: Há sobreposição entre treino e teste!")
    else:
        print(f"   ✅ Separação treino/teste correta (sem sobreposição)")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 5: Treinamento do Modelo
# ============================================================================
print("\n🤖 ETAPA 5: Treinando modelo...")
try:
    model = train_model(X_train, y_train)
    print(f"   ✅ Modelo treinado com sucesso")
    print(f"   ✅ Tipo: {type(model).__name__}")
    print(f"   ✅ N. árvores: {model.n_estimators}")
    print(f"   ✅ Features utilizadas: {model.n_features_in_}")
    print(f"   ✅ Score no treino: {model.score(X_train, y_train):.4f}")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 6: Predições
# ============================================================================
print("\n📊 ETAPA 6: Realizando predições...")
try:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"   ✅ Predições realizadas")
    print(f"   ✅ Predições (valores únicos): {np.unique(y_pred)}")
    print(f"   ✅ Distribuição de predições:")
    print(f"      - Classe 0: {(y_pred == 0).sum()}")
    print(f"      - Classe 1: {(y_pred == 1).sum()}")
    print(f"   ✅ Probabilidades (min, max, média):")
    print(f"      - Min: {y_pred_proba.min():.4f}")
    print(f"      - Max: {y_pred_proba.max():.4f}")
    print(f"      - Média: {y_pred_proba.mean():.4f}")
    
    # Verificar consistência
    y_pred_2 = model.predict(X_test)
    if np.array_equal(y_pred, y_pred_2):
        print(f"   ✅ Predições consistentes (determinísticas)")
    else:
        print(f"   ❌ Predições não-determinísticas!")
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 7: Avaliação do Modelo
# ============================================================================
print("\n📈 ETAPA 7: Avaliando modelo...")
try:
    evaluator = evaluate_model(
        y_test,
        y_pred,
        y_pred_proba,
        class_names=['Sem Risco de Defasagem', 'Com Risco de Defasagem'],
        plot_all=False
    )
    
    print(f"\n   ✅ Avaliação concluída")
    
    # Métricas esperadas
    metrics = evaluator.metrics
    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    print(f"\n   📊 Métricas Calculadas:")
    for metric in required_metrics:
        if metric in metrics:
            print(f"      ✅ {metric}: {metrics[metric]:.4f}")
        else:
            print(f"      ❌ {metric}: FALTANDO")
    
    # Matriz de confusão
    cm = evaluator.get_confusion_matrix()
    print(f"\n   🔍 Matriz de Confusão:")
    print(f"      TN (Verdadeiros Negativos): {cm[0, 0]}")
    print(f"      FP (Falsos Positivos):      {cm[0, 1]}")
    print(f"      FN (Falsos Negativos):      {cm[1, 0]}")
    print(f"      TP (Verdadeiros Positivos): {cm[1, 1]}")
    
    # DataFrame de métricas
    metrics_df = evaluator.to_dataframe()
    print(f"\n   📋 Exportação para DataFrame:")
    print(f"      ✅ Shape: {metrics_df.shape}")
    print(f"      ✅ Colunas: {list(metrics_df.columns)}")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# ETAPA 8: Persistência (Salvar/Carregar)
# ============================================================================
print("\n💾 ETAPA 8: Testando persistência...")
try:
    # Criar diretório temporário
    temp_dir = Path('temp_test_models')
    temp_dir.mkdir(exist_ok=True)
    
    model_path = temp_dir / 'test_model.joblib'
    features_path = temp_dir / 'test_features.joblib'
    
    # Salvar modelo
    save_model(model, str(model_path))
    print(f"   ✅ Modelo salvo em: {model_path}")
    
    # Salvar features
    joblib.dump(X_train.columns.tolist(), str(features_path))
    print(f"   ✅ Features salvas em: {features_path}")
    
    # Carregar modelo
    loaded_model = joblib.load(str(model_path))
    print(f"   ✅ Modelo carregado com sucesso")
    
    # Carregar features
    loaded_features = joblib.load(str(features_path))
    print(f"   ✅ Features carregadas: {len(loaded_features)} features")
    
    # Verificar predições são idênticas
    y_pred_loaded = loaded_model.predict(X_test)
    if np.array_equal(y_pred, y_pred_loaded):
        print(f"   ✅ Predições idênticas após reload")
    else:
        print(f"   ❌ Predições mudaram após reload!")
    
    # Exportar métricas para CSV
    metrics_csv_path = temp_dir / 'metrics.csv'
    metrics_df.to_csv(str(metrics_csv_path))
    print(f"   ✅ Métricas exportadas para: {metrics_csv_path}")
    
    # Limpeza
    import shutil
    shutil.rmtree(temp_dir)
    print(f"   ✅ Diretório temporário removido")
    
except Exception as e:
    print(f"   ❌ ERRO: {e}")
    sys.exit(1)

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 70)
print("✨ TESTE COMPLETO FINALIZADO COM SUCESSO! ✨")
print("=" * 70)

print("\n📋 RESUMO DO QUE FOI TESTADO:")
print("   ✅ 1. Carregamento e criação de dados")
print("   ✅ 2. Preprocessamento de dados")
print("   ✅ 3. Feature engineering")
print("   ✅ 4. Split treino/teste")
print("   ✅ 5. Treinamento do modelo")
print("   ✅ 6. Predições (classe + probabilidade)")
print("   ✅ 7. Avaliação com todas as métricas")
print("   ✅ 8. Persistência (salvar/carregar)")

print("\n📊 RESULTADO DAS MÉTRICAS:")
print(f"   • Acurácia:     {metrics['accuracy']:.4f}")
print(f"   • Precisão:     {metrics['precision']:.4f}")
print(f"   • Recall:       {metrics['recall']:.4f}")
print(f"   • F1-Score:     {metrics['f1_score']:.4f}")
print(f"   • ROC-AUC:      {metrics['roc_auc']:.4f}")

print("\n" + "=" * 70)
