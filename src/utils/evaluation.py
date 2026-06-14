import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, cohen_kappa_score, 
    matthews_corrcoef, confusion_matrix
)

def evaluate_and_plot(y_true, y_pred, y_prob, train_losses, train_accuracies, output_dir="../outputs"):
    """
    Calcula as métricas avançadas e salva os gráficos de evidência visual.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cálculos Estatísticos
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') # Macro dá o mesmo peso para UP e DOWN
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print("\n" + "="*50)
    print("🏆 RESULTADOS AVANÇADOS DE VALIDAÇÃO 🏆")
    print("="*50)
    print(f"Acurácia Geral : {acc*100:.2f}%")
    print(f"F1-Score Macro : {f1:.4f} (Equilíbrio Precisão/Recall)")
    print(f"ROC-AUC        : {roc_auc:.4f} (Capacidade de distinção)")
    print(f"PR-AUC         : {pr_auc:.4f} (Foco na classe minoritária)")
    print(f"Kappa Score    : {kappa:.4f} (Acertos além da sorte)")
    print(f"MCC            : {mcc:.4f} (Métrica suprema)")
    print("="*50 + "\n")

    # 2. Gráfico 1: Curvas de Aprendizado (Loss e Acurácia)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Treino Loss', color='red')
    plt.title('Evolução da Perda (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Treino Acurácia', color='blue')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

    # 3. Gráfico 2: Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Realidade')
    plt.xlabel('Previsão do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    print(f"✅ Gráficos de evidência visual salvos na pasta: {output_dir}")