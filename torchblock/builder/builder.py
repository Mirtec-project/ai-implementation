import json
import torch
import torch.nn as nn
from torchblock.builder.base_model import BaseModel
from torchblock.layers import LAYER_TYPES

class ModelBuilder:
    """JSON 설정 파일로부터 PyTorch 모델을 생성하는 빌더 클래스"""
    
    @classmethod
    def from_config(cls, config_path):
        """설정 파일로부터 모델을 생성"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls.build_model(config)
    
    @classmethod
    def build_model(cls, config):
        """설정 딕셔너리로부터 모델을 생성"""
        layers = []
        
        for layer_config in config['layers']:
            layer_type = layer_config['type']
            
            if layer_type in LAYER_TYPES:
                layer_class = LAYER_TYPES[layer_type]
                layer_args = layer_config.get('args', {})
                layers.append(layer_class(**layer_args))
        
        return BaseModel(config['model_name'], nn.Sequential(*layers))



def create_model(config_path):
    """편의를 위한 모델 생성 함수"""
    return ModelBuilder.from_config(config_path)
