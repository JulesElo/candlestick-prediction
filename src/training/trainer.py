import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Adiciona a raiz do projeto ao sys.path para importação do módulo src
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG
from src.data.dataset import ChronologicalDataset
from src.training.evaluation import evaluate_and_plot

def train_walk_forward():
    """
    Executa o treinamento e a validação do modelo utilizando fatiamento temporal (Walk-Forward).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de processamento: {device}")

    if CONFIG.model.normalization_type == "imagenet":
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        norm_mean, norm_std = CONFIG.model.mean, CONFIG.model.std

    transform = transforms.Compose([
        transforms.Resize((CONFIG.model.image_size, CONFIG.model.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    full_dataset = ChronologicalDataset(transform=transform)
    tscv = TimeSeriesSplit(n_splits=CONFIG.training.n_splits)
    
    fold_metrics = {'acc': [], 'f1': [], 'roc_auc': [], 'pr_auc': [], 'kappa': [], 'mcc': []}

    for fold, (train_index, test_index) in enumerate(tscv.split(full_dataset)):
        print(f"\n[{fold+1}/{CONFIG.training.n_splits}] Treino: {len(train_index)} | Teste: {len(test_index)}")

        train_subset = Subset(full_dataset, train_index)
        test_subset = Subset(full_dataset, test_index)

        train_loader = DataLoader(train_subset, batch_size=CONFIG.training.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=CONFIG.training.batch_size, shuffle=False)

        # Instanciação dinâmica da arquitetura selecionada no CONFIG
        if CONFIG.model.model_name == "resnet":
            from src.models.resnet import CandlestickResNet
            model = CandlestickResNet().to(device)
        else:
            from src.models.cnn import CandlestickCNN
            model = CandlestickCNN().to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)
        
        # Reduz a taxa de aprendizado pela metade a cada 10 épocas para estabilizar a convergência
        if CONFIG.training.use_lr_decay:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []

        for epoch in range(CONFIG.training.epochs):
            # Ativa modo de treinamento (habilita cálculo de gradientes e Dropout)
            model.train()
            train_loss, correct_train, total_train = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Zera os gradientes residuais da iteração anterior
                optimizer.zero_grad()
                
                # Forward pass: calcula as predições
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass: calcula as derivadas parciais do erro
                loss.backward()
                
                # Otimização: ajusta os pesos da rede
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_acc = 100 * correct_train / total_train
            history_train_loss.append(train_loss / len(train_loader))
            history_train_acc.append(train_acc)

            # Ativa modo de avaliação (desabilita Dropout)
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            
            # Desabilita o motor de diferenciação automática para economizar memória e processamento
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_acc = 100 * correct_val / total_val
            history_val_loss.append(val_loss / len(test_loader))
            history_val_acc.append(val_acc)

            print(f"Epoca [{epoch+1}/{CONFIG.training.epochs}] | Loss: {train_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(test_loader):.4f} - Acc: {val_acc:.2f}%")

            if CONFIG.training.use_lr_decay:
                scheduler.step()

        model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Converte os logits brutos em probabilidades entre 0 e 1
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs[:, 1].cpu().numpy())

        fold_dir = CONFIG.paths.experiments_dir / f"fold_{fold+1}"
        metrics = evaluate_and_plot(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob, 
            train_losses=history_train_loss, val_losses=history_val_loss, 
            train_accuracies=history_train_acc, val_accuracies=history_val_acc, 
            output_dir=fold_dir
        )
        
        fold_metrics['acc'].append(metrics['acc'])
        fold_metrics['f1'].append(metrics['f1'])
        fold_metrics['roc_auc'].append(metrics['roc_auc'])
        fold_metrics['pr_auc'].append(metrics['pr_auc'])
        fold_metrics['kappa'].append(metrics['kappa'])
        fold_metrics['mcc'].append(metrics['mcc'])

    print(f"\nRESUMO FINAL - {CONFIG.training.n_splits} FOLDS")
    print(f"Acurácia Média: {np.mean(fold_metrics['acc'])*100:.2f}%")
    print(f"F1-Score Médio: {np.mean(fold_metrics['f1']):.4f}")
    print(f"ROC-AUC Médio : {np.mean(fold_metrics['roc_auc']):.4f}")
    print(f"PR-AUC Médio  : {np.mean(fold_metrics['pr_auc']):.4f}")
    print(f"Kappa Médio   : {np.mean(fold_metrics['kappa']):.4f}")
    print(f"MCC Médio     : {np.mean(fold_metrics['mcc']):.4f}")

if __name__ == "__main__":
    try:
        train_walk_forward()
    except Exception as e:
        print(f"Erro de execução: {e}")