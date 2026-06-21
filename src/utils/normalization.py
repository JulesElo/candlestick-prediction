from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import CONFIG
from src.data.data_loader import ChronologicalDataset

def calculate_normalization_params(batch_size: int = 32) -> Tuple[List[float], List[float]]:
    """
    Calcula a média e o desvio padrão dos canais RGB das imagens do dataset.

    Args:
        batch_size (int): Tamanho do lote para leitura.

    Returns:
        Tuple[List[float], List[float]]: Médias e desvios padrão para [R, G, B].
    """
    transform = transforms.Compose([
        transforms.Resize((CONFIG.model.image_size, CONFIG.model.image_size)),
        transforms.ToTensor()
    ])

    dataset = ChronologicalDataset(transform=transform)
    
    if len(dataset) == 0:
        raise ValueError("Nenhuma imagem encontrada no dataset.")
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    channels_sum = torch.tensor([0.0, 0.0, 0.0])
    channels_squared_sum = torch.tensor([0.0, 0.0, 0.0])
    num_batches = 0
    
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    try:
        mean, std = calculate_normalization_params()
        print(f"\nmean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    except Exception as e:
        print(f"Erro ao calcular a normalização: {e}")