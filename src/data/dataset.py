import os
import sys
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

# Ajuste no caminho do sistema para importar o config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

class ChronologicalDataset(Dataset):
    """
    Dataset PyTorch customizado que lê todas as imagens de uma única pasta,
    garantindo a ordenação cronológica rigorosa e extraindo o rótulo do nome do arquivo.
    """
    def __init__(self, root_dir: Optional[str] = None, transform: Optional[Callable] = None):
        """
        Inicializa o leitor de dados temporais.

        Args:
            root_dir (str, optional): Caminho para a pasta com as imagens. 
                                      Se None, utiliza o caminho padrão do CONFIG.
            transform (Callable, optional): Transformações do torchvision a serem 
                                            aplicadas na imagem (ex: Resize, Normalize).
        """
        # Utiliza o caminho do config se nenhum for passado explicitamente
        self.root_dir = root_dir or CONFIG.paths.processed_data_dir
        self.transform = transform
        self.classes = ['down', 'up']
        self.class_to_idx = {'down': 0, 'up': 1}
        self.filepaths = []
        
        if os.path.exists(self.root_dir):
            for filename in os.listdir(self.root_dir):
                if filename.endswith(".png"):
                    # Extrai o rótulo do nome (ex: "20240101_to_20240130_up.png" -> "up")
                    label_str = filename.split('_')[-1].replace('.png', '')
                    label_idx = self.class_to_idx.get(label_str)
                    
                    if label_idx is not None:
                        self.filepaths.append((os.path.join(self.root_dir, filename), label_idx, filename))
        
        # Ordena a lista globalmente pelo nome do arquivo (que começa com a data YYYYMMDD)
        # Isso garante que a janela Walk-Forward não sofra vazamento temporal
        self.filepaths.sort(key=lambda x: x[2])
        
    def __len__(self) -> int:
        """Retorna o total de imagens no dataset."""
        return len(self.filepaths)
        
    def __getitem__(self, idx: int):
        """
        Busca uma imagem e seu rótulo na posição 'idx'.
        
        Args:
            idx (int): O índice cronológico da imagem.
            
        Returns:
            Tuple[torch.Tensor, int]: A imagem processada (Tensor) e seu rótulo (0 ou 1).
        """
        path, label, _ = self.filepaths[idx]
        
        # Converte explicitamente para RGB para evitar erros com imagens em tons de cinza
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label