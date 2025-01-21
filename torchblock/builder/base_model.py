import torch.nn as nn

class BaseModel(nn.Module):
    """생성된 레이어들을 포함하는 커스텀 모델 클래스"""
    
    def __init__(self, name, layers):
        super().__init__()
        self.name = name
        self.layers = layers
        
    def forward(self, x):
        return self.layers(x)