import torch
from torchblock.utils.iou import iou

def nms(bboxes, iou_threshold, prob_threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes
    
    Args:
        bboxes (list): List of bboxes where each bbox is [class_id, prob, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        prob_threshold (float): threshold to filter weak predictions
        box_format (str): "midpoint" or "corners" used to specify bbox format
        
    Returns:
        list: bboxes after performing NMS
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        
        remaining_boxes = []
        for box in bboxes:
            iou_value = iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            )
            # print(f"IOU between boxes - Class: {box[0]} vs {chosen_box[0]}, IOU: {float(iou_value):.4f}")
            
            if (box[0] != chosen_box[0]) or (iou_value < iou_threshold):
                # print(f"Box {box} is not removed because it is not the same class or IOU is greater than {iou_threshold}")
                remaining_boxes.append(box)
        
        bboxes = remaining_boxes
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

