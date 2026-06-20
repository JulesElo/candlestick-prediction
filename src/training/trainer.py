import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Ajuste de caminho para importar os módulos da raiz do src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from data.dataset import ChronologicalDataset
from models.cnn import CandlestickCNN
from models.resnet import CandlestickResNet

# Importa o motor de avaliação que agora mora na mesma pasta
from training.evaluation import evaluate_and_plot

def train_walk_forward():
    """
    Executa o treinamento e validação Walk-Forward da CNN utilizando as
    configurações centralizadas no arquivo config.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento com: {device}")

    # Logica de normalização dinâmica
    if CONFIG.model.normalization_type == "imagenet":
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        print("Aviso: Utilizando normalizacao padrao ImageNet.")
    else:
        norm_mean, norm_std = CONFIG.model.mean, CONFIG.model.std
        print("Aviso: Utilizando normalizacao customizada do dataset.")

    # Configura as transformações usando os valores do config
    transform = transforms.Compose([
        transforms.Resize((CONFIG.model.image_size, CONFIG.model.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # Carrega o dataset completo em ordem cronológica (sem passar caminhos, o config decide)
    full_dataset = ChronologicalDataset(transform=transform)
    print(f"Total de imagens cronológicas identificadas: {len(full_dataset)}")

    # O motor que corta a linha do tempo em fatias temporais
    tscv = TimeSeriesSplit(n_splits=CONFIG.training.n_splits)
    
    fold_metrics = {
        'acc': [],
        'f1': [],
        'roc_auc': [],
        'mcc': []
    }

    for fold, (train_index, test_index) in enumerate(tscv.split(full_dataset)):
        print("\n" + "="*50)
        print(f"INICIANDO FOLD {fold+1}/{CONFIG.training.n_splits}")
        print(f"Treino: {len(train_index)} imagens | Teste: {len(test_index)} imagens")
        print("="*50)

        # Separa os dados deste fold específico
        train_subset = Subset(full_dataset, train_index)
        test_subset = Subset(full_dataset, test_index)

        train_loader = DataLoader(train_subset, batch_size=CONFIG.training.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=CONFIG.training.batch_size, shuffle=False)

        # Instancia o modelo ZERADO para cada fold dinamicamente
        if CONFIG.model.model_name == "resnet":
            model = CandlestickResNet().to(device)
        else:
            model = CandlestickCNN().to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)
        
        if CONFIG.training.use_lr_decay:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []

        # ================= LOOP DE TREINAMENTO =================
        for epoch in range(CONFIG.training.epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_acc = 100 * correct_train / total_train
            history_train_loss.append(train_loss/len(train_loader))
            history_train_acc.append(train_acc)

            # Fase de Validação Rápida (para o gráfico)
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            
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
            history_val_loss.append(val_loss/len(test_loader))
            history_val_acc.append(val_acc)

            print(f"Época [{epoch+1}/{CONFIG.training.epochs}] | Treino Loss: {train_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(test_loader):.4f} - Acc: {val_acc:.2f}%")

            if CONFIG.training.use_lr_decay:
                scheduler.step()

        print(f"\nAvaliando dados de Teste do Fold {fold+1}...")

        # ================= AVALIAÇÃO DO FOLD =================
        model.eval()
        
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs[:, 1].cpu().numpy())

        # Salva os gráficos deste Fold na pasta centralizada do CONFIG
        fold_dir = os.path.join(CONFIG.paths.experiments_dir, f"fold_{fold+1}")
        metrics = evaluate_and_plot(
            y_true=y_true, 
            y_pred=y_pred, 
            y_prob=y_prob, 
            train_losses=history_train_loss, 
            val_losses=history_val_loss, 
            train_accuracies=history_train_acc, 
            val_accuracies=history_val_acc, 
            output_dir=fold_dir
        )
        
        # Guardando as métricas para fazer a média no final de todos os folds
        fold_metrics['acc'].append(metrics['acc'])
        fold_metrics['f1'].append(metrics['f1'])
        fold_metrics['roc_auc'].append(metrics['roc_auc'])
        fold_metrics['mcc'].append(metrics['mcc'])

    # ================= FIM DE TODOS OS FOLDS =================
    # Cálculo da Média Final (Fora do loop de folds)
    print(f"\n{'*'*50}\nRESUMO FINAL - MÉDIA DOS {CONFIG.training.n_splits} FOLDS\n{'*'*50}")
    print(f"Acurácia Média : {np.mean(fold_metrics['acc'])*100:.2f}%")
    print(f"F1-Score Médio   : {np.mean(fold_metrics['f1']):.4f}")
    print(f"ROC-AUC Médio  : {np.mean(fold_metrics['roc_auc']):.4f}")
    print(f"MCC Médio      : {np.mean(fold_metrics['mcc']):.4f}")
    print('*'*50)

if __name__ == "__main__":
    try:
        train_walk_forward()
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")