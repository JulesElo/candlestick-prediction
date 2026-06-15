import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import TimeSeriesSplit

# Ajuste de caminho para importar os módulos da raiz do src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from data.dataset import ChronologicalDataset
from models.cnn import CandlestickCNN

# Importa o motor de avaliação que agora mora na mesma pasta
from training.evaluation import evaluate_and_plot

def train_walk_forward():
    """
    Executa o treinamento e validação Walk-Forward da CNN utilizando as
    configurações centralizadas no arquivo config.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento Walk-Forward utilizando: {device}")

    # Configura as transformações usando os valores do config
    transform = transforms.Compose([
        transforms.Resize((CONFIG.model.image_size, CONFIG.model.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG.model.mean, std=CONFIG.model.std)
    ])

    # Carrega o dataset completo em ordem cronológica (sem passar caminhos, o config decide)
    full_dataset = ChronologicalDataset(transform=transform)
    print(f"Total de imagens cronológicas identificadas: {len(full_dataset)}")

    # O motor que corta a linha do tempo em fatias temporais
    tscv = TimeSeriesSplit(n_splits=CONFIG.training.n_splits)

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

        # Instancia o modelo ZERADO para cada fold
        model = CandlestickCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.training.learning_rate)
        
        if CONFIG.training.use_lr_decay:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        history_losses = []
        history_accuracies = []

        # ================= LOOP DE TREINAMENTO =================
        for epoch in range(CONFIG.training.epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            history_losses.append(running_loss/len(train_loader))
            history_accuracies.append(train_accuracy)

            print(f"Época [{epoch+1}/{CONFIG.training.epochs}] - Perda: {running_loss/len(train_loader):.4f} - Acurácia Treino: {train_accuracy:.2f}%")

            if CONFIG.training.use_lr_decay:
                scheduler.step()

        print(f"\nAvaliando dados de Teste do Fold {fold+1}...")

        # ================= AVALIAÇÃO DO FOLD =================
        model.eval()
        
        all_true_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_true_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs[:, 1].cpu().numpy())

        # Salva os gráficos deste Fold na pasta centralizada do CONFIG
        fold_output_dir = os.path.join(CONFIG.paths.experiments_dir, f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        evaluate_and_plot(
            y_true=all_true_labels, 
            y_pred=all_predictions, 
            y_prob=all_probabilities, 
            train_losses=history_losses, 
            train_accuracies=history_accuracies,
            output_dir=fold_output_dir
        )

if __name__ == "__main__":
    try:
        train_walk_forward()
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")