import os
import sys
import torch
import torch.nn as nn
from torchvision import models

# Ajuste no caminho para importar o config central
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

class CandlestickResNet(nn.Module):
    """
    Arquitetura baseada em Transfer Learning utilizando a ResNet18.
    A rede aproveita os pesos pré-treinados para extração profunda de 
    características visuais, adaptando apenas a camada de classificação final.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Inicializa a rede residual.

        Args:
            pretrained (bool, optional): Se True, baixa os pesos da ImageNet. 
                                         Padrão é True.
        """
        super(CandlestickResNet, self).__init__()
        
        # Carrega a arquitetura oficial. O parâmetro weights substitui o antigo 'pretrained=True'
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # ==========================================
        # O CONGELAMENTO DOS PESOS (FREEZING)
        # ==========================================
        # Varre todos os parâmetros da rede e impede que o otimizador os altere
        if pretrained:
            for name, param in self.resnet.named_parameters():
                # Descongela a camada 4 (último bloco convolucional) e a camada fully connected
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Extrai o número de neurônios da camada final original da ResNet
        num_ftrs = self.resnet.fc.in_features
        
        # Substitui a cabeça de classificação (1000 classes da ImageNet) pelas nossas 2 classes
        # Adicionamos um Dropout severo para ajudar a combater o ruído do mercado financeiro
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fluxo de dados através da ResNet."""
        return self.resnet(x)

if __name__ == "__main__":
    print("Instanciando a ResNet18 para validação de sanidade da arquitetura...")
    dummy_input = torch.randn(1, 3, CONFIG.model.image_size, CONFIG.model.image_size)
    model = CandlestickResNet()
    output = model(dummy_input)
    print(f"Sucesso! Dimensão do output esperado: [1, 2] -> Output real: {list(output.shape)}")