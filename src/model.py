import torch
import torch.nn as nn
import torch.nn.functional as F

class CandlestickCNN(nn.Module):
    """
    Arquitetura da Rede Neural Convolucional baseada no artigo de referência.
    Recebe imagens de 224x224 pixels e classifica em UP (Alta) ou DOWN (Baixa).
    """
    def __init__(self):
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
        
        # Matemática da redução de dimensionalidade (MaxPooling):
        # Imagem original: 224x224
        # Após pool1: 112x112
        # Após pool2: 56x56
        # Após pool3: 28x28
        # Após pool4: 14x14
        # Portanto, a saída da última convolução terá 96 canais de 14x14 pixels.
        
        # Camada densa (Fully Connected)
        # O flatten transforma os tensores 3D em um vetor 1D
        self.fc1 = nn.Linear(96 * 14 * 14, 512)
        self.drop3 = nn.Dropout(p=0.5)
        
        # Camada de saída: 2 neurônios (Classe 0: DOWN, Classe 1: UP)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        """
        Define o fluxo (passagem para frente) dos dados pela rede.
        O F.relu aplica a função de ativação ReLU em cada camada convolucional.
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop2(self.pool4(F.relu(self.conv4(x))))
        
        # Achatar (Flatten) a matriz para alimentar a camada Densa
        x = x.view(-1, 96 * 14 * 14)
        
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        
        return x

# Teste rápido para validar se as dimensões da rede estão corretas
if __name__ == "__main__":
    # Cria uma imagem "falsa" (tensor) com as dimensões esperadas (Batch=1, Canais=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Instancia o modelo
    model = CandlestickCNN()
    
    # Passa a imagem pela rede
    output = model(dummy_input)
    
    print("Modelo instanciado com sucesso!")
    print(f"Formato de entrada: {dummy_input.shape}")
    print(f"Formato de saída: {output.shape} (Esperado: 1 batch, 2 classes)")