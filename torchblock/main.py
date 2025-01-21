import torch
from torchblock.builder import create_model

def get_device():
    """CUDA 사용 가능 여부에 따라 장치를 반환"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(config_path):
    """JSON 설정 파일로부터 모델을 생성하고 출력"""
    model = create_model(config_path)
    print(model)
    return model

def test_model(model, input_shape):
    """모델을 테스트하고 출력 형태를 확인"""
    input_data = torch.randn(*input_shape)
    output = model(input_data)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    device = get_device()
    print(device)

    # JSON 설정 파일 경로
    config_path = 'model1.json'

    # 모델 생성 및 출력
    model = load_model(config_path)

    # 모델 테스트
    test_model(model, (1, 1, 32, 32))  # 1채널, 32x32 이미지