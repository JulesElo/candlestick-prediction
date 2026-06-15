import os
from typing import Tuple, List
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import TimeSeriesSplit

# Importa o motor de cálculos de métricas
from utils.evaluation import evaluate_and_plot

# Importa a arquitetura da CNN que criamos no arquivo model.py
from model import CandlestickCNN

class ChronologicalDataset(Dataset):
    """
    Dataset que lê todas as imagens de uma única pasta, garantindo
    a ordenação cronológica e extraindo o rótulo do nome do arquivo.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['down', 'up']
        self.class_to_idx = {'down': 0, 'up': 1}
        self.filepaths = []
        
        # Varre a pasta única "all"
        if os.path.exists(root_dir):
            for filename in os.listdir(root_dir):
                if filename.endswith(".png"):
                    # Extrai o rótulo do nome (ex: "20240101_to_20240130_up.png" -> "up")
                    label_str = filename.split('_')[-1].replace('.png', '')
                    label_idx = self.class_to_idx[label_str]
                    
                    self.filepaths.append((os.path.join(root_dir, filename), label_idx, filename))
        
        # Ordena a lista globalmente pelo nome do arquivo (que começa com a data YYYYMMDD)
        self.filepaths.sort(key=lambda x: x[2])
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        path, label, _ = self.filepaths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    mean: List[float] = [0.0, 0.0, 0.0],
    std: List[float] = [1.0, 1.0, 1.0],
    use_scheduler: bool = True,
    n_splits: int = 5
) -> None:
    """
    Orquestra o treinamento e a avaliação da CNN utilizando Validação Cruzada Walk-Forward.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento com TimeSeriesSplit utilizando: {device}")

    # Transformações com a normalização exata recebida do painel de controle
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Carrega todo o dataset em ordem cronológica
    dataset_path = os.path.join(data_dir, "all")
    full_dataset = ChronologicalDataset(root_dir=dataset_path, transform=transform)
    print(f"Total de imagens cronológicas identificadas: {len(full_dataset)}")

    # O motor que corta a linha do tempo em fatias
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_index, test_index) in enumerate(tscv.split(full_dataset)):
        print("\n" + "="*50)
        print(f"INICIANDO FOLD {fold+1}/{n_splits}")
        print(f"Treino: {len(train_index)} imagens | Teste: {len(test_index)} imagens")
        print("="*50)

        # Separa os dados deste fold específico
        train_subset = Subset(full_dataset, train_index)
        test_subset = Subset(full_dataset, test_index)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Instancia o modelo ZERADO para cada fold (evita vazamento de memória passada)
        model = CandlestickCNN(image_size=image_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        if use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        history_losses = []
        history_accuracies = []

        # ================= LOOP DE TREINAMENTO DO FOLD =================
        for epoch in range(epochs):
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

            print(f"Época [{epoch+1}/{epochs}] - Perda: {running_loss/len(train_loader):.4f} - Acurácia Treino: {train_accuracy:.2f}%")

            if use_scheduler:
                scheduler.step()

        print(f"\nAvaliando dados de Teste do Fold {fold+1}...")

        # ================= AVALIAÇÃO DO FOLD =================
        model.eval()
        
        # Variáveis para a conferência interna do terminal
        correct_test = 0
        total_test = 0

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

                # Contagem interna para exibir no terminal (Acurácia simples)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print(f"=> [CONFERÊNCIA INTERNA] Acurácia Final no Fold {fold+1}: {test_accuracy:.2f}%")

        # Salva os gráficos deste Fold em uma pasta dedicada
        fold_output_dir = os.path.join("..", "outputs", f"fold_{fold+1}")
        evaluate_and_plot(
            y_true=all_true_labels, 
            y_pred=all_predictions, 
            y_prob=all_probabilities, 
            train_losses=history_losses, 
            train_accuracies=history_accuracies,
            output_dir=fold_output_dir
        )

if __name__ == "__main__":
    # =========================================================================
    # PAINEL DE CONTROLE DE EXPERIMENTOS
    # Parâmetros ajustáveis para novos testes sem alterar o código
    # =========================================================================
    DATA_DIRECTORY = os.path.join("..", "images")
    
    # Parâmetros atuais (Configuração do EXP-08)
    IMAGE_SIZE = 224
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    USE_LR_DECAY = True
    N_SPLITS = 5 # Quantidade de janelas temporais da Validação Cruzada
    
    # Normalização
    MEAN = [0.0395, 0.0198, 0.0]
    STD = [0.1803, 0.0905, 1.0]

    try:
        train_model(
            data_dir=DATA_DIRECTORY,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            mean=MEAN,
            std=STD,
            use_scheduler=USE_LR_DECAY,
            n_splits=N_SPLITS
        )
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")