"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
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
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        # Darknet(Convolution 부분) 생성
        self.darknet = self._create_conv_layers(self.architecture)
        # FC 부분은 나중에 동적으로 만들기 위해 일단 None 처리
        self.fcs = None

        # split_size, num_boxes, num_classes 보관(나중에 fcs 생성 시 사용)
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # 더미 입력 한 번 통과시켜서 feature map 크기 확인 후 FC 레이어 생성
        self._create_fcs_after_build()

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                kernel_size, out_channels, stride, pad = x
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad)
                )
                in_channels = out_channels
            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(x) == list:
                conv1, conv2, num_repeats = x
                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    )
                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs_after_build(self):
        """
        Darknet으로부터 최종 feature map 크기를 동적으로 계산하고,
        그에 맞춰 fully-connected 레이어를 생성한다.
        """
        # 더미 입력(배치=1, 채널=3, 크기=448x448)을 ConvNet 통과해보고 feature 맵 크기를 확인
        dummy_x = torch.randn(1, self.in_channels, 448, 448)
        with torch.no_grad():
            out = self.darknet(dummy_x)
        flattened_size = out.flatten(start_dim=1).shape[1]

        # 예시로 hidden 노드를 4096개로 두고 싶다면:
        hidden_dim = 496

        # 최종 예측 레이어 출력 채널: S*S*(C + B*5)
        final_out_dim = self.S * self.S * (self.C + self.B * 5)

        self.fcs = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, final_out_dim),
        )

def test(split_size=7, num_boxes=2, num_classes=20):
    model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(model(x).shape)


if __name__ == "__main__":
    test()
