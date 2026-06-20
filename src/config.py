from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class PathsConfig:
    """Configurações de diretórios do projeto."""
    
    base_dir: Path = Path(__file__).resolve().parent.parent 
    raw_data_dir: Path = base_dir / "data" / "raw"
    processed_data_dir: Path = base_dir / "data" / "processed" / "images"
    experiments_dir: Path = base_dir / "experiments"

@dataclass
class ModelConfig:
    """Configurações da arquitetura da rede neural."""

    model_name: str = "cnn"  # "cnn" ou "resnet"
    image_size: int = 224
    normalization_type: str = "custom"  # "custom" ou "imagenet"
    mean: List[float] = field(default_factory=lambda: [0.0395, 0.0198, 0.0])
    std: List[float] = field(default_factory=lambda: [0.1803, 0.0905, 1.0])

@dataclass
class TrainingConfig:
    """Configurações dos hiperparâmetros de treinamento."""
    
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.0001
    use_lr_decay: bool = False
    n_splits: int = 5  # Número de janelas para a validação Walk-Forward

@dataclass
class ProjectConfig:
    """Agregação central de todas as classes de configuração."""
    
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

CONFIG = ProjectConfig()