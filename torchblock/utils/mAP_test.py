import torch
import pytest
from torchblock.utils.mAP import mAP

def test_mAP_perfect_prediction():
    """완벽한 예측일 경우의 mAP 테스트"""
    pred_boxes = [
        [0, 1, 0.9, 0, 0, 10, 10],  # [이미지_인덱스, 클래스, 확률, x1, y1, x2, y2]
        [0, 0, 0.8, 20, 20, 30, 30],
        [1, 1, 0.95, 40, 40, 50, 50]
    ]
    
    true_boxes = [
        [0, 1, 1.0, 0, 0, 10, 10],  # [이미지_인덱스, 클래스, 확률, x1, y1, x2, y2]
        [0, 0, 1.0, 20, 20, 30, 30],
        [1, 1, 1.0, 40, 40, 50, 50]
    ]
    
    result = mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=2)
    assert abs(result - 1.0) < 1e-5

def test_mAP_no_predictions():
    """예측이 하나도 없는 경우의 테스트"""
    pred_boxes = []
    true_boxes = [
        [0, 0, 1.0, 0, 0, 10, 10],    # [이미지_인덱스, 클래스, 확률, x1, y1, x2, y2]
        [0, 1, 1.0, 20, 20, 30, 30]
    ]
    
    result = mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=2)
    assert result == 0.0  # 예측이 없으므로 mAP는 0이어야 함

def test_mAP_partial_match():
    """일부만 정확한 예측의 경우 테스트"""
    pred_boxes = [
        [0, 0, 0.9, 0, 0, 10, 10],
        [0, 0, 0.8, 20, 20, 30, 30],
        [1, 1, 0.95, 40, 40, 50, 50]
    ]
    
    true_boxes = [
        [0, 0, 1.0, 0, 0, 10, 10],
        [0, 1, 1.0, 20, 20, 30, 30],
        [1, 1, 1.0, 40, 40, 50, 50]
    ]
    
    result = mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=2)
    assert 0.0 < result < 1.0  # mAP는 0과 1 사이여야 함

def test_mAP_different_iou_thresholds():
    """다른 IOU 임계값에 대한 테스트"""
    pred_boxes = [
        [0, 0, 0.9, 0, 0, 10, 10],
        [0, 0, 0.8, 19, 19, 31, 31],
    ]
    
    true_boxes = [
        [0, 0, 1.0, 0, 0, 10, 10],
        [0, 0, 1.0, 20, 20, 30, 30]
    ]
    
    result_strict = mAP(pred_boxes, true_boxes, iou_threshold=0.9, box_format="corners", num_classes=1)
    result_relaxed = mAP(pred_boxes, true_boxes, iou_threshold=0.3, box_format="corners", num_classes=1)
    
    assert result_strict < result_relaxed  # 더 엄격한 IOU 임계값에서는 mAP가 더 낮아야 함
