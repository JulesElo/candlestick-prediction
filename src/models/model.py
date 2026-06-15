import torch
import torch.nn as nn
import torch.nn.functional as F

class CandlestickCNN(nn.Module):
    """
    Arquitetura da Rede Neural Convolucional baseada no artigo de referência.
    Classifica imagens de candlesticks em UP (Alta) ou DOWN (Baixa).
    """
    def __init__(self, image_size: int = 224):
        """
        Inicializa as camadas da Rede Neural e calcula dinamicamente 
        as dimensões internas com base na resolução da imagem.

        Args:
            image_size (int, optional): A dimensão (largura e altura) da imagem quadrada 
                                        de entrada. Padrão é 224.
        """
        super(CandlestickCNN, self).__init__()
        
        # A imagem de entrada terá 3 canais (RGB)
        
        # 1ª Camada: Conv2D (32 filtros) + MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2ª Camada: Conv2D (48 filtros) + MaxPool + Dropout
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.25)
        
        # 3ª Camada: Conv2D (64 filtros) + MaxPool
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4ª Camada: Conv2D (96 filtros) + MaxPool + Dropout
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.25)
        
        # Matemática automatizada da redução de dimensionalidade (MaxPooling):
        # Como temos 4 camadas de MaxPooling (2x2), a imagem é reduzida pela metade 4 vezes.
        # Portanto, o tamanho linear da grade cai 2^4 = 16 vezes.
        self.final_grid_size = image_size // 16 
        self.flattened_size = 96 * self.final_grid_size * self.final_grid_size
        
        # Camada densa (Fully Connected)
        # O flatten transforma os tensores 3D em um vetor 1D
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.drop3 = nn.Dropout(p=0.5)
        
        # Camada de saída: 2 neurônios (Classe 0: DOWN, Classe 1: UP)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define o fluxo (passagem para frente) dos dados pela rede.
        O F.relu aplica a função de ativação ReLU em cada camada convolucional.

        Args:
            x (torch.Tensor): Tensor contendo o lote de imagens.

        Returns:
            torch.Tensor: Tensor com as previsões numéricas brutas (logits) para as 2 classes.
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop2(self.pool4(F.relu(self.conv4(x))))
        
        # Achatar (Flatten) a matriz para alimentar a camada Densa (usando o cálculo dinâmico do init)
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        
        return x

# Teste rápido para validar se as dimensões da rede estão corretas
if __name__ == "__main__":
    # Variável para testar rapidamente qualquer resolução antes de treinar
    TEST_IMAGE_SIZE = 224 
    
    # Cria uma imagem "falsa" (tensor) com as dimensões esperadas
    dummy_input = torch.randn(1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
    
    # Instancia o modelo passando a resolução correta
    model = CandlestickCNN(image_size=TEST_IMAGE_SIZE)
    
    # Passa a imagem pela rede
    output = model(dummy_input)
    
    print("Modelo instanciado e testado com sucesso!")
    print(f"Formato de entrada: {dummy_input.shape}")
    print(f"Formato de saída: {output.shape} (Esperado: 1 batch, 2 classes)")
    print(f"Cálculo interno do Flatten: {model.flattened_size} neurônios.")