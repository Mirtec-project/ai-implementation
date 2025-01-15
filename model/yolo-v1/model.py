import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    # 첫 번째 레이어: 7x7 커널, 64개 필터, stride 2, padding 3
    # 큰 커널 사이즈로 이미지의 전반적인 특징을 잡아냄
    (7, 64, 2, 3),
    
    # 2x2 MaxPooling, stride 2
    "M",
    
    # 3x3 커널, 192개 필터로 특징 추출
    (3, 192, 1, 1),
    
    # 2x2 MaxPooling, stride 2 
    "M",
    
    # 1x1 컨볼루션으로 채널 수 감소 (차원 축소)
    (1, 128, 1, 0),
    # 3x3 컨볼루션으로 특징 추출
    (3, 256, 1, 1),
    # 1x1 컨볼루션으로 채널 조정
    (1, 256, 1, 0),
    # 3x3 컨볼루션으로 특징 추출, 채널 수 증가
    (3, 512, 1, 1),
    
    # 2x2 MaxPooling, stride 2
    "M",
    
    # 4번 반복되는 블록: 1x1 축소 -> 3x3 확장 구조
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    
    # 1x1 컨볼루션으로 채널 조정
    (1, 512, 1, 0),
    # 3x3 컨볼루션으로 특징 추출, 채널 수 증가
    (3, 1024, 1, 1),
    
    # 2x2 MaxPooling, stride 2
    "M",
    
    # 2번 반복되는 블록: 1x1 축소 -> 3x3 확장 구조
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    
    # 마지막 레이어들: 3x3 컨볼루션들로 최종 특징 추출
    # 깊은 레이어에서 고수준 특징들을 잡아냄
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += CNNBlock(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                )
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list:
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2] # Integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # * 원본 논문에서는 4096 - 시간이 오래걸리기 때문에 임시로 줄임
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # (S, S, 30) where C+B*5 = 30
        )