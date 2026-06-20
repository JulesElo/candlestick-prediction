import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adiciona a raiz do projeto ao sys.path para importação do módulo src
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import CONFIG

class CandlestickCNN(nn.Module):
    """
    Rede Neural Convolucional para classificação de imagens de candlesticks.
    """
    def __init__(self, image_size: int = CONFIG.model.image_size):
        """
        Inicializa as camadas e calcula o tamanho do vetor achatado (Flatten).

        Args:
            image_size (int): Dimensão de largura e altura da imagem de entrada.
        """
        super(CandlestickCNN, self).__init__()
        
        # Camada 1: Entrada [3, H, W] -> Saída [32, H, W]. Filtro 3x3 com padding=1 preserva H e W.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Reduz H e W pela metade. Ex: 224x224 torna-se 112x112.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camada 2: Entrada [32, H/2, W/2] -> Saída [48, H/2, W/2].
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)
        # Reduz H e W pela metade novamente. Ex: 112x112 torna-se 56x56.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)
        
        # Camada 3: Entrada [48, H/4, W/4] -> Saída [64, H/4, W/4].
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        # Reduz H e W pela metade novamente. Ex: 56x56 torna-se 28x28.
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camada 4: Entrada [64, H/8, W/8] -> Saída [96, H/8, W/8].
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        # Reduz H e W pela metade final. Ex: 28x28 torna-se 14x14.
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.25)
        
        # Passagem de um tensor fictício para determinar o tamanho matemático do Flatten.
        # Evita a necessidade de cálculo manual quando o tamanho da imagem muda.
        dummy_tensor = torch.zeros(1, 3, image_size, image_size)
        with torch.no_grad():
            x = self.pool1(self.conv1(dummy_tensor))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            self.flattened_size = x.numel()
            
        # Camadas totalmente conectadas para classificação final.
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o fluxo de encaminhamento (forward pass) dos dados na rede.

        Args:
            x (torch.Tensor): Lote de imagens de entrada no formato [Batch, 3, H, W].

        Returns:
            torch.Tensor: Logits brutos de saída para as duas classes [Batch, 2].
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop2(self.pool4(F.relu(self.conv4(x))))
        
        # Converte a estrutura de matriz tridimensional [Canais, H, W] em um vetor linear.
        # O parâmetro -1 indica ao PyTorch que a dimensão do lote (Batch) deve ser preservada.
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    print(f"Testando instanciamento da CNN com resolução ({CONFIG.model.image_size})...")
    
    dummy_input = torch.randn(1, 3, CONFIG.model.image_size, CONFIG.model.image_size)
    model = CandlestickCNN()
    
    output = model(dummy_input)
    print(f"Sucesso. Formato do output real: {list(output.shape)}")