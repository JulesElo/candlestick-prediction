import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class PathsConfig:
    """Configuracoes de diretorios do projeto."""
    
    # Captura a raiz do projeto (duas pastas acima do config.py)
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_data_dir: str = os.path.join(base_dir, "data", "raw")
    processed_data_dir: str = os.path.join(base_dir, "data", "processed", "images")
    experiments_dir: str = os.path.join(base_dir, "experiments")

@dataclass
class ModelConfig:
    """Configuracoes da arquitetura da Rede Neural."""

    model_name: str = "resnet"  # Opções disponíveis: "cnn" ou "resnet"
    
    image_size: int = 224
    
    # Parametros de normalizacao calculados anteriormente (EXP-08)
    mean: List[float] = field(default_factory=lambda: [0.0395, 0.0198, 0.0])
    std: List[float] = field(default_factory=lambda: [0.1803, 0.0905, 1.0])

@dataclass
class TrainingConfig:
    """Configuracoes dos hiperparametros de treinamento."""
    
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    use_lr_decay: bool = True
    n_splits: int = 5  # Numero de janelas (folds) para a Validacao Walk-Forward

@dataclass
class ProjectConfig:
    """Classe principal que agrega todas as configuracoes."""
    
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# Instancia global para ser importada por outros modulos (ex: from config import CONFIG)
CONFIG = ProjectConfig()