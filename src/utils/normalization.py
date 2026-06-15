import os
import sys
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Ajuste no caminho do sistema para importar os módulos da raiz do src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from data.dataset import ChronologicalDataset

def calculate_normalization_params(batch_size: int = 32) -> Tuple[List[float], List[float]]:
    """
    Lê todas as imagens processadas e calcula a média e o desvio padrão exatos 
    dos canais RGB para calibração fina do modelo.

    Args:
        batch_size (int, optional): Tamanho do lote para leitura. Padrão é 32.

    Returns:
        Tuple[List[float], List[float]]: Listas com médias e desvios para [R, G, B].
    """
    # IMPORTANTE: Aplicamos apenas o Resize e ToTensor. 
    # Não aplicamos Normalize, pois é justamente o que queremos descobrir!
    transform = transforms.Compose([
        transforms.Resize((CONFIG.model.image_size, CONFIG.model.image_size)),
        transforms.ToTensor()
    ])

    print("Carregando o dataset de imagens...")
    dataset = ChronologicalDataset(transform=transform)
    
    if len(dataset) == 0:
        raise ValueError("Nenhuma imagem encontrada. Execute o gerador de imagens primeiro.")
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    channels_sum = torch.tensor([0.0, 0.0, 0.0])
    channels_squared_sum = torch.tensor([0.0, 0.0, 0.0])
    num_batches = 0

    print(f"Calculando Média e Desvio Padrão para imagens {CONFIG.model.image_size}x{CONFIG.model.image_size}. Aguarde...")
    
    for data, _ in tqdm(loader):
        # data = [lote, canais(RGB), altura, largura]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    try:
        mean, std = calculate_normalization_params()
        
        print("\n" + "="*50)
        print("=== Resultados da Normalização ===")
        print("="*50)
        print(f"Copie e cole estes valores no seu arquivo src/config.py:\n")
        print(f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Ocorreu um erro ao calcular a normalização: {e}")