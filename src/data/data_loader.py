from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

from src.config import CONFIG

class ChronologicalDataset(Dataset):
    """
    Dataset PyTorch customizado para leitura de imagens cronológicas.
    """
    def __init__(self, root_dir: Optional[Path] = None, transform: Optional[Callable] = None):
        """
        Inicializa o leitor de dados temporais.

        Args:
            root_dir (Path, optional): Caminho para a pasta com as imagens.
            transform (Callable, optional): Transformações torchvision aplicáveis.
        """
        self.root_dir = root_dir or CONFIG.paths.processed_data_dir
        self.transform = transform
        self.classes = ['down', 'up']
        self.class_to_idx = {'down': 0, 'up': 1}
        self.filepaths = []
        
        if self.root_dir.exists():
            for file_path in self.root_dir.glob("*.png"):
                label_str = file_path.stem.split('_')[-1]
                label_idx = self.class_to_idx.get(label_str)
                
                if label_idx is not None:
                    self.filepaths.append((file_path, label_idx, file_path.name))
        
        self.filepaths.sort(key=lambda x: x[2])
        
    def __len__(self) -> int:
        """Retorna o total de amostras no dataset."""
        return len(self.filepaths)
        
    def __getitem__(self, idx: int):
        """
        Carrega imagem e rótulo na posição 'idx'.
        
        Args:
            idx (int): Índice da imagem.
            
        Returns:
            Tuple[torch.Tensor, int]: Imagem processada e rótulo (0 ou 1).
        """
        path, label, _ = self.filepaths[idx]
        
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label