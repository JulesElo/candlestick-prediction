import torch
import torch.nn as nn
from torchvision import models

from src.config import CONFIG

class CandlestickResNet(nn.Module):
    """
    Arquitetura de classificação baseada na ResNet18 adaptada para duas classes.
    """
    def __init__(self, pretrained: bool = True):
        """
        Inicializa a rede ResNet18 e substitui a camada fully connected final.

        Args:
            pretrained (bool): Se True, utiliza os pesos padrão da ImageNet.
        """
        super(CandlestickResNet, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        num_ftrs = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encaminha os dados pela arquitetura ResNet18.

        Args:
            x (torch.Tensor): Lote de imagens de entrada.

        Returns:
            torch.Tensor: Logits de saída.
        """
        return self.resnet(x)

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, CONFIG.model.image_size, CONFIG.model.image_size)
    model = CandlestickResNet()
    output = model(dummy_input)
    print(f"Formato do output real: {list(output.shape)}")