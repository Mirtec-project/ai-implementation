import torch
import pytest
from torchblock.utils.iou import iou

def test_iou_midpoint_format():
    # 간단한 케이스: 완전히 겹치는 박스
    boxes_preds = torch.tensor([[0.5, 0.5, 1.0, 1.0]])  # (x,y,w,h)
    boxes_labels = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    result = iou(boxes_preds, boxes_labels, box_format="midpoint")
    print(f"완전히 겹치는 박스 IOU (midpoint): {result.item():.6f}")
    assert torch.allclose(result, torch.tensor([1.0]))

    # 부분적으로 겹치는 박스
    boxes_preds = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    boxes_labels = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    result = iou(boxes_preds, boxes_labels, box_format="midpoint")
    print(f"부분적으로 겹치는 박스 IOU (midpoint): {result.item():.6f}")
    expected_iou = torch.tensor([0.14285714])
    assert torch.allclose(result, expected_iou, rtol=1e-5)

    # 겹치지 않는 박스
    boxes_preds = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes_labels = torch.tensor([[2.0, 2.0, 1.0, 1.0]])
    assert torch.allclose(iou(boxes_preds, boxes_labels, box_format="midpoint"), torch.tensor([0.0]))

def test_iou_corners_format():
    # 완전히 겹치는 박스
    boxes_preds = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # (x1,y1,x2,y2)
    boxes_labels = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    result = iou(boxes_preds, boxes_labels, box_format="corners")
    print(f"완전히 겹치는 박스 IOU (corners): {result.item():.6f}")
    assert torch.allclose(result, torch.tensor([1.0]))

    # 부분적으로 겹치는 박스
    boxes_preds = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes_labels = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
    result = iou(boxes_preds, boxes_labels, box_format="corners")
    print(f"부분적으로 겹치는 박스 IOU (corners): {result.item():.6f}")
    expected_iou = torch.tensor([0.14285714])
    assert torch.allclose(result, expected_iou, rtol=1e-5)

    # 겹치지 않는 박스
    boxes_preds = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes_labels = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    assert torch.allclose(iou(boxes_preds, boxes_labels, box_format="corners"), torch.tensor([0.0]))

def test_iou_batch():
    # 배치 처리 테스트
    boxes_preds = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [0.5, 0.5, 1.0, 1.0]
    ])
    boxes_labels = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.5, 1.5]
    ])
    result = iou(boxes_preds, boxes_labels, box_format="corners")
    print(f"배치 처리 IOU 결과:\n{result}")
    assert result.shape == (2, 1)
    assert torch.allclose(result[0], torch.tensor([1.0]))

def test_invalid_box_format():
    boxes_preds = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes_labels = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    with pytest.raises(UnboundLocalError):
        iou(boxes_preds, boxes_labels, box_format="invalid_format") 