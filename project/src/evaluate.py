"""
Módulo de Avaliação de Modelos de Machine Learning.

Este módulo fornece funções para calcular e visualizar métricas de desempenho
de modelos de classificação, incluindo:
- Métricas básicas (Acurácia, Precisão, Recall, F1-Score)
- Matriz de Confusão
- Curva ROC e AUC
- Análise detalhada por classe
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import pandas as pd
from typing import Dict, Tuple, Optional


class ModelEvaluator:
    """
    Classe para avaliação completa de modelos de classificação.
    
    Calcula e visualiza métricas de desempenho para modelos treinados,
    facilitando a análise de resultados e comparação entre modelos.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: Optional[np.ndarray] = None,
                 class_names: Optional[list] = None):
        """
        Inicializa o avaliador de modelo.
        
        Args:
            y_true: Array de labels verdadeiros
            y_pred: Array de predições (0/1 para classificação binária)
            y_pred_proba: Array de probabilidades preditas (opcional)
            class_names: Lista com nomes das classes (ex: ['Sem Risco', 'Com Risco'])
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class_names = class_names or ['Classe 0', 'Classe 1']
        self.metrics = {}
        
    def calculate_metrics(self) -> Dict:
        """
        Calcula todas as métricas de avaliação.
        
        Returns:
            Dicionário contendo todas as métricas calculadas
        """
        self.metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        
        # Calcular ROC-AUC se probabilidades forem fornecidas
        if self.y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            self.metrics['average_precision'] = average_precision_score(
                self.y_true, self.y_pred_proba
            )
        
        return self.metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Calcula a matriz de confusão.
        
        Returns:
            Matriz de confusão (2x2 para classificação binária)
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self) -> str:
        """
        Gera relatório detalhado de classificação.
        
        Returns:
            String contendo o relatório de classificação
        """
        return classification_report(
            self.y_true, 
            self.y_pred,
            target_names=self.class_names,
            digits=4
        )
    
    def print_summary(self):
        """Imprime um resumo das métricas calculadas."""
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "="*60)
        print("RESUMO DE MÉTRICAS DO MODELO")
        print("="*60)
        
        print(f"\n📊 MÉTRICAS PRINCIPAIS:")
        print(f"  • Acurácia:        {self.metrics['accuracy']:.4f}")
        print(f"  • Precisão:        {self.metrics['precision']:.4f}")
        print(f"  • Recall (Sensib.): {self.metrics['recall']:.4f}")
        print(f"  • F1-Score:        {self.metrics['f1_score']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"  • ROC-AUC:         {self.metrics['roc_auc']:.4f}")
        if 'average_precision' in self.metrics:
            print(f"  • AP (Avg Prec):   {self.metrics['average_precision']:.4f}")
        
        print(f"\n🔍 MATRIZ DE CONFUSÃO:")
        cm = self.get_confusion_matrix()
        print(f"  TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
        
        # Calcular taxa de verdadeiros positivos e negativos
        total = cm.sum()
        tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n📈 MÉTRICAS DERIVADAS:")
        print(f"  • Especificidade:  {specificity:.4f}")
        print(f"  • Sensibilidade:   {sensitivity:.4f}")
        
        print(f"\n📋 RELATÓRIO DETALHADO:")
        print(self.get_classification_report())
    
    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (8, 6),
                              cmap: str = 'Blues', save_path: Optional[str] = None):
        """
        Plota a matriz de confusão.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
            cmap: Mapa de cores do matplotlib
            save_path: Caminho para salvar a figura (opcional)
        """
        cm = self.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar_kws={'label': 'Frequência'},
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_ylabel('Verdadeiro', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predito', fontsize=12, fontweight='bold')
        ax.set_title('Matriz de Confusão', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Matriz de confusão salva em: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (8, 6),
                       save_path: Optional[str] = None):
        """
        Plota a curva ROC.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
            save_path: Caminho para salvar a figura (opcional)
        """
        if self.y_pred_proba is None:
            print("⚠️  Probabilidades não fornecidas. Impossível plotar curva ROC.")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plotar curva ROC
        ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        
        # Plotar diagonal (classificador aleatório)
        ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
                label='Random Classifier')
        
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=11, fontweight='bold')
        ax.set_title('Curva ROC', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Curva ROC salva em: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, figsize: Tuple[int, int] = (8, 6),
                                   save_path: Optional[str] = None):
        """
        Plota a curva Precisão-Recall.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
            save_path: Caminho para salvar a figura (opcional)
        """
        if self.y_pred_proba is None:
            print("⚠️  Probabilidades não fornecidas. Impossível plotar curva Precisão-Recall.")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        ap = average_precision_score(self.y_true, self.y_pred_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, color='#2ca02c', lw=2.5,
                label=f'PR Curve (AP = {ap:.4f})')
        
        ax.set_xlabel('Recall (Sensibilidade)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Precisão', fontsize=11, fontweight='bold')
        ax.set_title('Curva Precisão-Recall', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Curva Precisão-Recall salva em: {save_path}")
        
        plt.show()
    
    def plot_all_metrics(self, figsize: Tuple[int, int] = (14, 10),
                         save_path: Optional[str] = None):
        """
        Plota todas as métricas em um único dashboard.
        
        Args:
            figsize: Tamanho da figura (largura, altura)
            save_path: Caminho para salvar a figura (opcional)
        """
        if not self.metrics:
            self.calculate_metrics()
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Matriz de Confusão
        ax1 = fig.add_subplot(gs[0, 0])
        cm = self.get_confusion_matrix()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Frequência'})
        ax1.set_title('Matriz de Confusão', fontweight='bold')
        ax1.set_ylabel('Verdadeiro')
        ax1.set_xlabel('Predito')
        
        # 2. Métricas em barras
        ax2 = fig.add_subplot(gs[0, 1])
        metrics_to_plot = {
            'Acurácia': self.metrics['accuracy'],
            'Precisão': self.metrics['precision'],
            'Recall': self.metrics['recall'],
            'F1-Score': self.metrics['f1_score']
        }
        if 'roc_auc' in self.metrics:
            metrics_to_plot['ROC-AUC'] = self.metrics['roc_auc']
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax2.bar(range(len(metrics_to_plot)), list(metrics_to_plot.values()),
                       color=colors[:len(metrics_to_plot)])
        ax2.set_xticks(range(len(metrics_to_plot)))
        ax2.set_xticklabels(metrics_to_plot.keys(), rotation=45, ha='right')
        ax2.set_ylabel('Score')
        ax2.set_ylim([0, 1.05])
        ax2.set_title('Métricas Principais', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Curva ROC
        if self.y_pred_proba is not None:
            ax3 = fig.add_subplot(gs[1, 0])
            fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'AUC = {roc_auc:.4f}')
            ax3.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
            ax3.set_xlabel('Taxa de Falsos Positivos')
            ax3.set_ylabel('Taxa de Verdadeiros Positivos')
            ax3.set_title('Curva ROC', fontweight='bold')
            ax3.legend(loc='lower right')
            ax3.grid(alpha=0.3)
        
        # 4. Curva Precisão-Recall
        if self.y_pred_proba is not None:
            ax4 = fig.add_subplot(gs[1, 1])
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
            ap = average_precision_score(self.y_true, self.y_pred_proba)
            ax4.plot(recall, precision, color='#2ca02c', lw=2.5, label=f'AP = {ap:.4f}')
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precisão')
            ax4.set_title('Curva Precisão-Recall', fontweight='bold')
            ax4.legend(loc='best')
            ax4.grid(alpha=0.3)
        
        plt.suptitle('Dashboard de Avaliação do Modelo', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Dashboard salvo em: {save_path}")
        
        plt.show()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte as métricas em um DataFrame.
        
        Returns:
            DataFrame contendo as métricas
        """
        if not self.metrics:
            self.calculate_metrics()
        
        cm = self.get_confusion_matrix()
        tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics_extended = {
            **self.metrics,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        return pd.DataFrame([metrics_extended]).T.rename(columns={0: 'Valor'})


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_pred_proba: Optional[np.ndarray] = None,
                   class_names: Optional[list] = None,
                   plot_all: bool = True) -> ModelEvaluator:
    """
    Função auxiliar para avaliação rápida de modelos.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        y_pred_proba: Probabilidades (opcional)
        class_names: Nomes das classes (opcional)
        plot_all: Se True, plota todas as métricas
    
    Returns:
        Instância de ModelEvaluator com métricas calculadas
    """
    evaluator = ModelEvaluator(y_true, y_pred, y_pred_proba, class_names)
    evaluator.calculate_metrics()
    evaluator.print_summary()
    
    if plot_all and y_pred_proba is not None:
        evaluator.plot_all_metrics()
    
    return evaluator


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from preprocessing import load_data, preprocess_data, split_data
    from feature_engineering import apply_feature_engineering
    from train import train_model
    
    # Carregar e preparar dados
    print("📂 Carregando dados...")
    data_path = 'data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx'
    try:
        df = load_data(data_path)
        df_processed = preprocess_data(df)
        df_engineered = apply_feature_engineering(df_processed)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = split_data(df_engineered)
        
        # Treinar modelo
        print("🤖 Treinando modelo...")
        model = train_model(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Avaliar modelo
        print("\n📊 Avaliando modelo...")
        evaluator = evaluate_model(
            y_test, 
            y_pred, 
            y_pred_proba,
            class_names=['Sem Risco', 'Com Risco'],
            plot_all=True
        )
        
        # Salvar resultados como CSV
        metrics_df = evaluator.to_dataframe()
        metrics_df.to_csv('model_metrics.csv')
        print("\n✅ Métricas salvas em 'model_metrics.csv'")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
