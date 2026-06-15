import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ajuste no caminho para importar o config central
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

class CandlestickCNN(nn.Module):
    """
    Arquitetura da Rede Neural Convolucional (CNN) baseada na literatura 
    (Kusuma et al., 2019) para classificação de imagens de candlesticks 
    em tendência de Alta (UP) ou Baixa (DOWN).
    """
    
    def __init__(self, image_size: int = CONFIG.model.image_size):
        """
        Inicializa as camadas da Rede Neural e calcula dinamicamente 
        as dimensões internas para a camada Densa (Fully Connected) 
        com base na resolução de entrada configurada.

        Args:
            image_size (int, optional): A dimensão (largura/altura) da imagem 
                                        quadrada de entrada. Padrão vem do CONFIG.
        """
        super(CandlestickCNN, self).__init__()
        
        # 1ª Camada: Extração de características de baixo nível
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2ª Camada: Extração de padrões intermediários com regularização (Dropout)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)
        
        # 3ª Camada: Padrões complexos
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4ª Camada: Padrões de alto nível
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.25)
        
        # ==========================================
        # Cálculo Dinâmico do Flatten
        # ==========================================
        # Simula a passagem de um tensor vazio pela rede para descobrir
        # qual será o tamanho matemático da matriz antes de achatá-la.
        dummy_tensor = torch.zeros(1, 3, image_size, image_size)
        with torch.no_grad():
            x = self.pool1(self.conv1(dummy_tensor))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            self.flattened_size = x.numel()
            
        # ==========================================
        # Camadas Densas (Classificação Final)
        # ==========================================
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.drop3 = nn.Dropout(p=0.5)
        
        # Saída com 2 neurônios (Classe 0: DOWN, Classe 1: UP)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define o fluxo de dados (Forward Pass) através da rede.

        Args:
            x (torch.Tensor): Tensor contendo o lote de imagens.

        Returns:
            torch.Tensor: Tensor com as previsões numéricas brutas (logits).
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop2(self.pool4(F.relu(self.conv4(x))))
        
        # Achata (Flatten) a matriz 3D para um vetor 1D
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    print(f"Testando instanciamento da CNN com a resolução padrão do projeto ({CONFIG.model.image_size}x{CONFIG.model.image_size})...")
    
    # Cria uma imagem \"falsa\" apenas para validar a arquitetura
    dummy_input = torch.randn(1, 3, CONFIG.model.image_size, CONFIG.model.image_size)
    model = CandlestickCNN()
    
    output = model(dummy_input)
    print(f"Sucesso! Dimensão do output esperado: [1, 2] -> Output real: {list(output.shape)}")