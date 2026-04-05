import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Importa a arquitetura da CNN que criamos no arquivo model.py
from model import CandlestickCNN

def get_data_loaders(data_dir: str, batch_size: int = 32):
    """
    Prepara as imagens para o PyTorch.
    Converte as imagens para Tensores e normaliza os valores dos pixels.
    """
    # Transformações necessárias para preparar a imagem para a CNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Garantia de segurança para o tamanho
        transforms.ToTensor(),         # Converte a imagem (pixels de 0-255) para Tensor (0.0 a 1.0)
        transforms.Normalize(          # Normalização padrão usada em redes RGB
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Aponta para as pastas de treino e teste
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # O ImageFolder lê as pastas automaticamente e atribui as classes (down=0, up=1)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # DataLoaders criam lotes (batches) e embaralham os dados de treino
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Não embaralhamos os dados de teste para manter a avaliação consistente
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

def train_model():
    """
    Função principal que orquestra o treinamento da Rede Neural.
    """
    # 1. Hiperparâmetros (Configurações do treinamento)
    BATCH_SIZE = 32
    EPOCHS = 10         # Quantas vezes a rede verá todos os dados de treino
    LEARNING_RATE = 0.0001 # O tamanho do "passo" que a rede dá para aprender

    # Verifica se há uma placa de vídeo da Nvidia (CUDA) disponível para acelerar, senão usa o Processador (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento utilizando: {device}")

    # 2. Carrega os dados
    data_dir = os.path.join("..", "images")
    train_loader, test_loader, classes = get_data_loaders(data_dir, BATCH_SIZE)
    print(f"Classes identificadas nas pastas: {classes}")

    # Conta a distribuição real das classes no treino
    targets = train_loader.dataset.targets
    count_down = targets.count(0)
    count_up = targets.count(1)
    total = len(targets)
    print(f"Distribuição no Treino: DOWN={count_down} ({count_down/total*100:.2f}%) | UP={count_up} ({count_up/total*100:.2f}%)")

    # 3. Instancia o modelo e envia para a memória (CPU ou GPU)
    model = CandlestickCNN().to(device)

    # 4. Define a Função de Perda e o Otimizador
    # CrossEntropyLoss é o padrão ouro para problemas de classificação de múltiplas classes ou binária
    criterion = nn.CrossEntropyLoss()
    
    # Adam é o otimizador que ajustará os pesos da rede para minimizar a perda
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. O Loop de Treinamento
    for epoch in range(EPOCHS):
        model.train() # Coloca o modelo em modo de treinamento (ativa o Dropout)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Passa por todos os lotes (batches) de imagens
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zera os gradientes (ajustes) do lote anterior
            optimizer.zero_grad()

            # Forward pass: A rede tenta prever as classes das imagens
            outputs = model(images)
            
            # Calcula o erro (Loss) comparando a previsão com o rótulo real
            loss = criterion(outputs, labels)
            
            # Backward pass: Calcula matematicamente onde a rede errou
            loss.backward()
            
            # Otimização: Ajusta os filtros da rede baseados no erro calculado
            optimizer.step()

            # Estatísticas para exibir no terminal
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calcula a acurácia média da época atual
        train_accuracy = 100 * correct_train / total_train
        print(f"Época [{epoch+1}/{EPOCHS}] - Perda: {running_loss/len(train_loader):.4f} - Acurácia Treino: {train_accuracy:.2f}%")

    print("\nTreinamento concluído. Iniciando avaliação nos dados de Teste (que a rede nunca viu)...")

    # 6. Avaliação final (Teste)
    model.eval() # Coloca o modelo em modo de avaliação (desativa o Dropout para estabilidade)
    correct_test = 0
    total_test = 0

    # torch.no_grad() desliga o cálculo de gradientes para economizar memória e processamento
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f"\n=> Acurácia Final no conjunto de Teste: {test_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()