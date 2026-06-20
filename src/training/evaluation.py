import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, cohen_kappa_score, 
    matthews_corrcoef, confusion_matrix, roc_curve
)

def evaluate_and_plot(
    y_true: List[int], 
    y_pred: List[int], 
    y_prob: List[float], 
    train_losses: List[float],
    val_losses: List[float], 
    train_accuracies: List[float],
    val_accuracies: List[float],
    output_dir: Path
) -> Dict[str, float]:
    """
    Calcula métricas de validação e gera gráficos de desempenho do modelo.

    Args:
        y_true (List[int]): Rótulos reais das classes (0 ou 1).
        y_pred (List[int]): Predições discretas do modelo.
        y_prob (List[float]): Probabilidades contínuas para a classe 1.
        train_losses (List[float]): Histórico de perda de treino.
        val_losses (List[float]): Histórico de perda de validação.
        train_accuracies (List[float]): Histórico de acurácia de treino.
        val_accuracies (List[float]): Histórico de acurácia de validação.
        output_dir (Path): Diretório de destino para exportação dos gráficos.

    Returns:
        Dict[str, float]: Dicionário com as métricas calculadas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        'acc': acc,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'kappa': kappa,
        'mcc': mcc
    }
    
    print("\n" + "="*50)
    print("RESULTADOS DE VALIDAÇÃO")
    print("="*50)
    print(f"Acurácia Geral : {acc*100:.2f}%")
    print(f"F1-Score       : {f1:.4f}")
    print(f"ROC-AUC        : {roc_auc:.4f}")
    print(f"PR-AUC         : {pr_auc:.4f}")
    print(f"Kappa Score    : {kappa:.4f}")
    print(f"MCC            : {mcc:.4f}")
    print("="*50 + "\n")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Treino', color='blue')
    plt.plot(val_losses, label='Validação', color='orange', linestyle='--')
    plt.title('Evolução da Perda (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Treino', color='blue')
    plt.plot(val_accuracies, label='Validação', color='orange', linestyle='--')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Realidade')
    plt.xlabel('Previsão do Modelo')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    return metrics