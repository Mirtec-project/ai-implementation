"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # [N, S, S, C + B*5] -> [N, S, S, 2 + 2*5] = 12
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # 클래스가 2개일 때 인덱스 설정
        # [0..1]: 클래스(2개)
        # [2]: Box1 confidence
        # [3..6]: Box1 x,y,w,h
        # [7]: Box2 confidence
        # [8..11]: Box2 x,y,w,h
        box1_idx = self.C + 1  # 예: 3
        box2_idx = self.C + 6  # 예: 8
        obj_idx = self.C       # 예: 2 (Box1 confidence 채널을 존재 여부로 활용)

        # IoU 계산
        iou_b1 = intersection_over_union(
            predictions[..., box1_idx:box1_idx+4],
            target[..., box1_idx:box1_idx+4]
        )
        iou_b2 = intersection_over_union(
            predictions[..., box2_idx:box2_idx+4],
            target[..., box1_idx:box1_idx+4]
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        exists_box = target[..., obj_idx].unsqueeze(3)  # Iobj_i (해당 셀에 객체가 있으면 1, 없으면 0)

        # ======================== #
        #     BOX COORDINATES     #
        # ======================== #
        box_predictions = exists_box * (
            bestbox * predictions[..., box2_idx:box2_idx+4]
            + (1 - bestbox) * predictions[..., box1_idx:box1_idx+4]
        )
        box_targets = exists_box * target[..., box1_idx:box1_idx+4]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            box_predictions.view(-1, 4),
            box_targets.view(-1, 4)
        )

        # ==================== #
        #    OBJECT LOSS      #
        # ==================== #
        # responsible box 의 confidence 예측
        pred_box_conf = (
            bestbox * predictions[..., box2_idx-1:box2_idx]
            + (1 - bestbox) * predictions[..., self.C:self.C+1]
        )
        object_loss = self.mse(
            (exists_box * pred_box_conf).view(-1),
            (exists_box * target[..., obj_idx:obj_idx+1]).view(-1),
        )

        # ======================= #
        #   NO OBJECT LOSS       #
        # ======================= #
        # 각 셀에서 객체가 없을 때 confidence 손실
        no_object_loss = self.mse(
            ((1 - exists_box) * predictions[..., self.C:self.C+1]).view(predictions.shape[0], -1),
            ((1 - exists_box) * target[..., obj_idx:obj_idx+1]).view(predictions.shape[0], -1),
        )
        no_object_loss += self.mse(
            ((1 - exists_box) * predictions[..., box2_idx-1:box2_idx]).view(predictions.shape[0], -1),
            ((1 - exists_box) * target[..., obj_idx:obj_idx+1]).view(predictions.shape[0], -1),
        )

        # ================== #
        #     CLASS LOSS    #
        # ================== #
        # 객체가 있는 셀에 대해서만 클래스 예측 손실 계산
        class_loss = self.mse(
            (exists_box * predictions[..., :self.C]).view(-1, self.C),
            (exists_box * target[..., :self.C]).view(-1, self.C),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
