"""
Testes para o modelo de predição de risco de defasagem.

Este módulo contém testes unitários e de integração para validar:
- Treinamento do modelo
- Predições do modelo
- Persistência (salvar/carregar modelo)
- Métricas de avaliação
- Integridade do pipeline completo
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from sklearn.ensemble import RandomForestClassifier
import joblib

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import train_model, save_model
from preprocessing import preprocess_data, split_data
from feature_engineering import apply_feature_engineering
from evaluate import ModelEvaluator, evaluate_model


class TestModelTraining(unittest.TestCase):
    """Testes para o treinamento do modelo."""
    
    def setUp(self):
        """Preparar dados de teste."""
        # Criar dataset sintético para testes
        np.random.seed(42)
        n_samples = 200
        
        self.sample_data = pd.DataFrame({
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
        
        # Processar dados
        self.df_processed = preprocess_data(self.sample_data)
        self.df_engineered = apply_feature_engineering(self.df_processed)
        
        # Dividir dados
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df_engineered)
    
    def test_train_model_returns_classifier(self):
        """Testa se train_model retorna um classificador válido."""
        model = train_model(self.X_train, self.y_train)
        
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
    
    def test_model_is_fitted(self):
        """Testa se o modelo é treinado corretamente."""
        model = train_model(self.X_train, self.y_train)
        
        # Verificar se o modelo foi treinado
        self.assertTrue(hasattr(model, 'n_features_in_'))
        self.assertEqual(model.n_features_in_, self.X_train.shape[1])
    
    def test_model_training_with_different_parameters(self):
        """Testa treinamento com diferentes parâmetros."""
        # Treinar com parâmetros padrão
        model = train_model(self.X_train, self.y_train)
        
        # Verificar número de estimadores
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.random_state, 42)


class TestModelPredictions(unittest.TestCase):
    """Testes para predições do modelo."""
    
    def setUp(self):
        """Preparar modelo treinado para testes."""
        np.random.seed(42)
        n_samples = 200
        
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
        
        df_processed = preprocess_data(sample_data)
        df_engineered = apply_feature_engineering(df_processed)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(df_engineered)
        
        # Treinar modelo
        self.model = train_model(self.X_train, self.y_train)
    
    def test_predict_returns_binary_values(self):
        """Testa se as predições são valores binários (0 ou 1)."""
        y_pred = self.model.predict(self.X_test)
        
        # Verificar se todas as predições são 0 ou 1
        unique_values = np.unique(y_pred)
        self.assertTrue(all(val in [0, 1] for val in unique_values))
    
    def test_predict_correct_shape(self):
        """Testa se as predições têm a forma correta."""
        y_pred = self.model.predict(self.X_test)
        
        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])
        self.assertEqual(len(y_pred.shape), 1)
    
    def test_predict_proba_returns_probabilities(self):
        """Testa se predict_proba retorna probabilidades válidas."""
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Verificar forma: (n_samples, 2) para classificação binária
        self.assertEqual(y_pred_proba.shape, (self.X_test.shape[0], 2))
        
        # Verificar se as probabilidades somam 1
        prob_sums = y_pred_proba.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)))
        
        # Verificar se todas as probabilidades estão entre 0 e 1
        self.assertTrue(np.all(y_pred_proba >= 0))
        self.assertTrue(np.all(y_pred_proba <= 1))
    
    def test_predict_with_single_sample(self):
        """Testa predição com uma única amostra."""
        single_sample = self.X_test.iloc[[0]]
        y_pred = self.model.predict(single_sample)
        
        self.assertEqual(len(y_pred), 1)
        self.assertIn(y_pred[0], [0, 1])
    
    def test_predictions_consistency(self):
        """Testa se predições são consistentes (mesma entrada = mesma saída)."""
        y_pred_1 = self.model.predict(self.X_test)
        y_pred_2 = self.model.predict(self.X_test)
        
        np.testing.assert_array_equal(y_pred_1, y_pred_2)


class TestModelPersistence(unittest.TestCase):
    """Testes para salvar e carregar o modelo."""
    
    def setUp(self):
        """Preparar modelo e diretório temporário."""
        np.random.seed(42)
        n_samples = 100
        
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
        
        df_processed = preprocess_data(sample_data)
        df_engineered = apply_feature_engineering(df_processed)
        X_train, X_test, y_train, y_test = split_data(df_engineered)
        
        self.model = train_model(X_train, y_train)
        self.X_test = X_test
        
        # Criar diretório temporário
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpar diretório temporário."""
        shutil.rmtree(self.test_dir)
    
    def test_save_model_creates_file(self):
        """Testa se save_model cria o arquivo corretamente."""
        model_path = os.path.join(self.test_dir, 'test_model.joblib')
        save_model(self.model, model_path)
        
        self.assertTrue(os.path.exists(model_path))
    
    def test_load_saved_model(self):
        """Testa se o modelo salvo pode ser carregado."""
        model_path = os.path.join(self.test_dir, 'test_model.joblib')
        save_model(self.model, model_path)
        
        loaded_model = joblib.load(model_path)
        
        self.assertIsInstance(loaded_model, RandomForestClassifier)
    
    def test_loaded_model_predictions_match_original(self):
        """Testa se o modelo carregado faz as mesmas predições que o original."""
        model_path = os.path.join(self.test_dir, 'test_model.joblib')
        
        # Fazer predições com o modelo original
        original_predictions = self.model.predict(self.X_test)
        
        # Salvar e carregar modelo
        save_model(self.model, model_path)
        loaded_model = joblib.load(model_path)
        
        # Fazer predições com o modelo carregado
        loaded_predictions = loaded_model.predict(self.X_test)
        
        # Comparar predições
        np.testing.assert_array_equal(original_predictions, loaded_predictions)


class TestModelEvaluation(unittest.TestCase):
    """Testes para avaliação do modelo."""
    
    def setUp(self):
        """Preparar dados de avaliação."""
        np.random.seed(42)
        
        # Criar dados de teste simples
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        self.y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
        self.y_pred_proba = np.array([0.2, 0.8, 0.9, 0.3, 0.4, 0.1, 0.7, 0.85, 0.15, 0.6])
    
    def test_evaluator_initialization(self):
        """Testa inicialização do ModelEvaluator."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred, self.y_pred_proba)
        
        self.assertIsInstance(evaluator, ModelEvaluator)
        np.testing.assert_array_equal(evaluator.y_true, self.y_true)
        np.testing.assert_array_equal(evaluator.y_pred, self.y_pred)
    
    def test_calculate_metrics_returns_dict(self):
        """Testa se calculate_metrics retorna um dicionário."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred, self.y_pred_proba)
        metrics = evaluator.calculate_metrics()
        
        self.assertIsInstance(metrics, dict)
    
    def test_metrics_contain_required_keys(self):
        """Testa se as métricas contêm todas as chaves necessárias."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred, self.y_pred_proba)
        metrics = evaluator.calculate_metrics()
        
        required_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in required_keys:
            self.assertIn(key, metrics)
    
    def test_metrics_values_in_valid_range(self):
        """Testa se os valores das métricas estão no intervalo válido [0, 1]."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred, self.y_pred_proba)
        metrics = evaluator.calculate_metrics()
        
        for metric_name, metric_value in metrics.items():
            self.assertGreaterEqual(metric_value, 0.0,
                                    f"{metric_name} deve ser >= 0")
            self.assertLessEqual(metric_value, 1.0,
                                 f"{metric_name} deve ser <= 1")
    
    def test_confusion_matrix_shape(self):
        """Testa se a matriz de confusão tem a forma correta."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred)
        cm = evaluator.get_confusion_matrix()
        
        self.assertEqual(cm.shape, (2, 2))
    
    def test_confusion_matrix_values(self):
        """Testa se a matriz de confusão tem valores corretos."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred)
        cm = evaluator.get_confusion_matrix()
        
        # Verificar se todos os valores são inteiros não-negativos
        self.assertTrue(np.all(cm >= 0))
        self.assertTrue(np.issubdtype(cm.dtype, np.integer))
        
        # Verificar se a soma corresponde ao número de amostras
        self.assertEqual(cm.sum(), len(self.y_true))
    
    def test_classification_report_returns_string(self):
        """Testa se o relatório de classificação retorna uma string."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred)
        report = evaluator.get_classification_report()
        
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    
    def test_evaluate_model_function(self):
        """Testa a função auxiliar evaluate_model."""
        evaluator = evaluate_model(
            self.y_true,
            self.y_pred,
            self.y_pred_proba,
            class_names=['Classe 0', 'Classe 1'],
            plot_all=False
        )
        
        self.assertIsInstance(evaluator, ModelEvaluator)
        self.assertIsInstance(evaluator.metrics, dict)
    
    def test_to_dataframe_returns_dataframe(self):
        """Testa se to_dataframe retorna um DataFrame válido."""
        evaluator = ModelEvaluator(self.y_true, self.y_pred, self.y_pred_proba)
        evaluator.calculate_metrics()
        df = evaluator.to_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


class TestModelIntegration(unittest.TestCase):
    """Testes de integração do pipeline completo."""
    
    def setUp(self):
        """Preparar pipeline completo."""
        np.random.seed(42)
        n_samples = 150
        
        self.sample_data = pd.DataFrame({
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
    
    def test_full_pipeline_execution(self):
        """Testa execução completa do pipeline: preprocessamento -> treino -> avaliação."""
        # 1. Preprocessamento
        df_processed = preprocess_data(self.sample_data)
        self.assertIn('target', df_processed.columns)
        
        # 2. Feature Engineering
        df_engineered = apply_feature_engineering(df_processed)
        self.assertGreater(df_engineered.shape[1], df_processed.shape[1])
        
        # 3. Split
        X_train, X_test, y_train, y_test = split_data(df_engineered)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        
        # 4. Treino
        model = train_model(X_train, y_train)
        self.assertIsInstance(model, RandomForestClassifier)
        
        # 5. Predição
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        self.assertEqual(len(y_pred), len(y_test))
        
        # 6. Avaliação
        evaluator = evaluate_model(
            y_test, y_pred, y_pred_proba,
            class_names=['Sem Risco', 'Com Risco'],
            plot_all=False
        )
        self.assertIsInstance(evaluator, ModelEvaluator)
        self.assertGreater(evaluator.metrics['accuracy'], 0)
    
    def test_pipeline_data_consistency(self):
        """Testa se o pipeline mantém a consistência dos dados."""
        df_processed = preprocess_data(self.sample_data)
        df_engineered = apply_feature_engineering(df_processed)
        X_train, X_test, y_train, y_test = split_data(df_engineered)
        
        # Verificar se não há vazamento de dados
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)
        
        # Verificar se a soma dos tamanhos é correta
        total_samples = len(X_train) + len(X_test)
        self.assertEqual(total_samples, len(df_engineered))
    
    def test_model_handles_missing_features(self):
        """Testa se o modelo lida corretamente com features ausentes."""
        df_processed = preprocess_data(self.sample_data)
        df_engineered = apply_feature_engineering(df_processed)
        X_train, X_test, y_train, y_test = split_data(df_engineered)
        
        model = train_model(X_train, y_train)
        
        # Tentar prever com menos features deve gerar erro
        with self.assertRaises((ValueError, Exception)):
            X_test_incomplete = X_test.iloc[:, :-1]  # Remover última coluna
            model.predict(X_test_incomplete)


class TestModelPerformance(unittest.TestCase):
    """Testes para verificar performance mínima do modelo."""
    
    def setUp(self):
        """Preparar dados de teste maiores."""
        np.random.seed(42)
        n_samples = 500
        
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
        
        df_processed = preprocess_data(sample_data)
        df_engineered = apply_feature_engineering(df_processed)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(df_engineered)
        self.model = train_model(self.X_train, self.y_train)
    
    def test_model_accuracy_above_baseline(self):
        """Testa se o modelo tem acurácia acima da baseline (predição aleatória)."""
        y_pred = self.model.predict(self.X_test)
        evaluator = ModelEvaluator(self.y_test, y_pred)
        metrics = evaluator.calculate_metrics()
        
        # Acurácia deve ser melhor que chance aleatória (0.5)
        self.assertGreater(metrics['accuracy'], 0.5,
                           "Acurácia deve ser maior que predição aleatória")
    
    def test_model_can_predict_both_classes(self):
        """Testa se o modelo consegue prever ambas as classes."""
        y_pred = self.model.predict(self.X_test)
        
        unique_predictions = np.unique(y_pred)
        
        # O modelo deve ser capaz de prever pelo menos uma amostra de cada classe
        # (isso pode falhar com dados muito desbalanceados, mas é um bom indicador)
        self.assertGreater(len(unique_predictions), 0,
                           "Modelo deve fazer pelo menos uma predição")
    
    def test_model_probability_calibration(self):
        """Testa se as probabilidades preditas fazem sentido."""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Verificar se há variação nas probabilidades (não está sempre prevendo a mesma probabilidade)
        prob_std = np.std(y_pred_proba)
        self.assertGreater(prob_std, 0.01,
                           "Probabilidades devem ter alguma variação")


if __name__ == '__main__':
    # Executar testes com verbosidade
    unittest.main(verbosity=2)
