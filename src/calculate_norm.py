import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def calculate_normalization_params(
    data_dir: str, 
    image_size: Tuple[int, int] = (224, 224), 
    batch_size: int = 32
) -> Tuple[List[float], List[float]]:
    """
    Lê todas as imagens do conjunto de Treino e calcula a média 
    e o desvio padrão exatos dos canais RGB.

    Args:
        data_dir (str): O caminho para a pasta raiz das imagens (que contém 'train' e 'test').
        image_size (Tuple[int, int], optional): O tamanho para redimensionar as imagens antes do cálculo. 
                                                Padrão é (224, 224).
        batch_size (int, optional): O tamanho do lote para carregar na RAM. Padrão é 32.

    Returns:
        Tuple[List[float], List[float]]: Duas listas contendo as médias e desvios padrões 
                                         para os canais [R, G, B].
    """
    # Transformação básica: garantir o tamanho correto e converter para Tensor (0.0 a 1.0)
    # IMPORTANTE: Não colocamos o 'Normalize' aqui, porque é justamente ele que queremos descobrir!
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Aponta para a pasta de Treino
    train_dir = os.path.join(data_dir, "train")
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    # DataLoader para carregar as imagens em lotes (economiza RAM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Variáveis para acumular as somas matemáticas
    channels_sum = 0.0
    channels_squared_sum = 0.0
    num_batches = 0

    print(f"Calculando média e desvio padrão para imagens {image_size}. Aguarde...")
    
    # O loop varre todas as imagens da pasta de treino
    for data, _ in tqdm(loader):
        # O tensor 'data' tem o formato: [lote, canais(RGB), altura, largura]
        # Queremos a média das cores (canais), então ignoramos o lote(0), a altura(2) e a largura(3)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    # Calcula a média final dividindo pela quantidade de lotes lidos
    mean = channels_sum / num_batches
    
    # Calcula o desvio padrão usando a fórmula da variância: std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    # Retorna os resultados convertidos para lista simples do Python
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    # Aponta para a pasta raiz das imagens
    data_directory = os.path.join("..", "images")
    
    try:
        mean, std = calculate_normalization_params(data_directory, image_size=(100, 100))
        
        print("\n=== Resultados da Normalização Exata do seu Projeto ===")
        print(f"MEAN (Média)       : {[round(x, 4) for x in mean]}")
        print(f"STD  (Desvio Padrão): {[round(x, 4) for x in std]}")
        print("========================================================\n")
        
    except Exception as e:
        print(f"Ocorreu um erro ao tentar ler as imagens: {e}")