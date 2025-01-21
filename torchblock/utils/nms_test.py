import torch
import pytest
from torchblock.utils.nms import nms

def test_nms_basic():
    # 기본적인 NMS 테스트
    boxes = [
        [1, 0.9, 0.0, 0.0, 1.0, 1.0],  # class_id, prob, x1, y1, x2, y2
        [1, 0.8, 0.1, 0.1, 1.1, 1.1],  # 첫 번째 박스와 많이 겹침 (제거되어야 함)
        [1, 0.7, 0.5, 0.5, 1.5, 1.5],  # 부분적으로 겹침 (IOU > 0.5면 제거)
        [2, 0.95, 2.0, 2.0, 3.0, 3.0],  # 다른 클래스
    ]
    
    result = nms(boxes, iou_threshold=0.5, prob_threshold=0.5, box_format="corners")
    assert len(result) == 3  # 겹치는 박스들이 제거되고 다른 클래스는 유지
    assert result[0][0:2] == [2, 0.95]  # 가장 높은 확률의 박스
    assert result[1][0:2] == [1, 0.9]   # 다른 클래스의 가장 높은 확률 박스

def test_nms_empty():
    # 빈 리스트 테스트
    result = nms([], iou_threshold=0.5, prob_threshold=0.5)
    assert result == []

def test_nms_prob_threshold():
    # 확률 임계값 테스트
    boxes = [
        [1, 0.9, 0.0, 0.0, 1.0, 1.0],
        [1, 0.4, 0.1, 0.1, 1.1, 1.1],  # prob_threshold 미만
        [1, 0.3, 0.5, 0.5, 1.5, 1.5],  # prob_threshold 미만
    ]
    
    result = nms(boxes, iou_threshold=0.5, prob_threshold=0.5)
    assert len(result) == 1
    assert result[0][1] == 0.9

def test_nms_midpoint_format():
    # midpoint 포맷 테스트
    boxes = [
        [1, 0.9, 0.5, 0.5, 1.0, 1.0],  # class_id, prob, cx, cy, w, h
        [1, 0.8, 0.55, 0.55, 1.0, 1.0],  # 많이 겹침
        [1, 0.7, 1.0, 1.0, 1.0, 1.0],  # 덜 겹침
    ]
    
    result = nms(boxes, iou_threshold=0.5, prob_threshold=0.5, box_format="midpoint")
    assert len(result) == 2
    assert result[0][1] == 0.9
    assert result[1][1] == 0.7

def test_nms_invalid_input():
    # 잘못된 입력 테스트
    with pytest.raises(AssertionError):
        nms(torch.tensor([]), iou_threshold=0.5, prob_threshold=0.5)  # list가 아닌 입력

def test_nms_high_iou_threshold():
    # 높은 IOU 임계값 테스트
    boxes = [
        [1, 0.9, 0.0, 0.0, 1.0, 1.0],
        [1, 0.8, 0.1, 0.1, 1.1, 1.1],
        [1, 0.7, 0.2, 0.2, 1.2, 1.2],
    ]
    
    result = nms(boxes, iou_threshold=0.9, prob_threshold=0.5, box_format="corners")
    assert len(result) == 3  # 높은 IOU 임계값으로 인해 모든 박스가 유지됨
