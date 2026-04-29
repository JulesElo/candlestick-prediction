import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Importa a arquitetura da CNN que criamos no arquivo model.py
from model import CandlestickCNN

def get_data_loaders(
    data_dir: str, 
    batch_size: int = 32, 
    image_size: int = 224,
    mean: List[float] = [0.0, 0.0, 0.0],
    std: List[float] = [1.0, 1.0, 1.0]
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Prepara as imagens para o PyTorch, redimensionando, convertendo para Tensores 
    e aplicando a normalização estatística.

    Args:
        data_dir (str): Caminho raiz contendo as pastas 'train' e 'test'.
        batch_size (int, optional): Quantidade de imagens por lote. Padrão é 32.
        image_size (int, optional): Tamanho (A x L) para redimensionar as imagens. Padrão é 224.
        mean (List[float], optional): Médias dos canais RGB para normalização.
        std (List[float], optional): Desvios padrões dos canais RGB para normalização.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: Lotes de treino, lotes de teste e lista de classes.
    """
    # Transformações necessárias para preparar a imagem para a CNN
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),    # Garantia de segurança para o tamanho
        transforms.ToTensor(),                          # Converte a imagem (pixels de 0-255) para Tensor (0.0 a 1.0)
        transforms.Normalize(mean=mean, std=std)        # Normalização padrão usada em redes RGB
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

def train_model(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.0001,
    mean: List[float] = [0.0, 0.0, 0.0],
    std: List[float] = [1.0, 1.0, 1.0],
    use_scheduler: bool = False
) -> None:
    """
    Orquestra o treinamento e a avaliação da Rede Neural Convolucional.

    Args:
        data_dir (str): Caminho raiz do dataset de imagens.
        image_size (int, optional): Resolução da imagem. Padrão é 224.
        batch_size (int, optional): Tamanho do lote. Padrão é 32.
        epochs (int, optional): Quantidade de épocas de treinamento. Padrão é 50.
        learning_rate (float, optional): Taxa de aprendizado inicial. Padrão é 0.0001.
        mean (List[float], optional): Valores de média para normalização.
        std (List[float], optional): Valores de desvio padrão para normalização.
        use_scheduler (bool, optional): Se True, reduz o learning rate pela metade a cada 10 épocas.
    """

    # Verifica se há uma placa de vídeo da Nvidia (CUDA) disponível para acelerar, senão usa o Processador (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treinamento utilizando: {device}")

    # Carrega os dados
    train_loader, test_loader, classes = get_data_loaders(data_dir, batch_size, image_size, mean, std)
    print(f"Classes identificadas nas pastas: {classes}")

    # Conta a distribuição real das classes no treino
    targets = train_loader.dataset.targets
    count_down = targets.count(0)
    count_up = targets.count(1)
    total = len(targets)
    print(f"Distribuição no Treino: DOWN={count_down} ({count_down/total*100:.2f}%) | UP={count_up} ({count_up/total*100:.2f}%)")

    # Instancia o modelo e envia para a memória (CPU ou GPU)
    model = CandlestickCNN(image_size=image_size).to(device)

    # Define Função de Perda, Otimizador e Agendador (opcional)
    criterion = nn.CrossEntropyLoss()
    
    # Adam é o otimizador que ajustará os pesos da rede para minimizar a perda
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Configura o decaimento: reduz a taxa de aprendizado pela metade (gamma=0.5) a cada 10 épocas
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # O Loop de Treinamento
    for epoch in range(epochs):
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
        print(f"Época [{epoch+1}/{epochs}] - Perda: {running_loss/len(train_loader):.4f} - Acurácia Treino: {train_accuracy:.2f}%")

        # Avisa o agendador que uma época passou, para ele atualizar a taxa de aprendizado se necessário
        if use_scheduler:
            scheduler.step()

    print("\nTreinamento concluído. Iniciando avaliação nos dados de Teste...")

    # Avaliação final (Teste)
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
    # =========================================================================
    # PAINEL DE CONTROLE DE EXPERIMENTOS
    # Parâmetros ajustáveis para novos testes sem alterar o código
    # =========================================================================
    DATA_DIRECTORY = os.path.join("..", "images")
    
    # Parâmetros atuais (Configuração do EXP-06)
    IMAGE_SIZE = 224
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    USE_LR_DECAY = False
    
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
            use_scheduler=USE_LR_DECAY
        )
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")