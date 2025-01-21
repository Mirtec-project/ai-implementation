import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError("box_format must be 'midpoint' or 'corners'")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bbox
                       specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    # Filter out boxes below threshold
    bboxes = [box for box in bboxes if box[1] > threshold]
    # Sort boxes by probability score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # Different class => keep
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bbox
                           specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar to pred_boxes except all are ground truth
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # filter bboxes only for class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # count the number of ground truth bboxes for each image
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort detections by confidence
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # ground truth bbox can only be matched once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # Add sentinel values for start
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Use trapz to integrate
        ap = torch.trapz(precisions, recalls)
        average_precisions.append(ap)

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, box_format="midpoint"):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        # box is [class, score, x, y, w, h] (if midpoint format)
        if box_format == "midpoint":
            x_mid, y_mid, w, h = box[2], box[3], box[4], box[5]
            upper_left_x = x_mid - w / 2
            upper_left_y = y_mid - h / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                w * width,
                h * height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        elif box_format == "corners":
            x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
            rect = patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        else:
            raise ValueError("box_format must be 'midpoint' or 'corners'")

        ax.add_patch(rect)

    plt.show()


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    S=14,
    B=2,
    C=2
):
    """
    Obtains bounding boxes from a model across a dataloader.
    
    Parameters:
        loader (DataLoader): PyTorch dataloader.
        model: Your PyTorch model.
        iou_threshold (float)
        threshold (float): threshold on confidence scores
        pred_format (str): "cells" if model outputs grid cell format, "other" if not
        box_format (str): "midpoint"/"corners"
        device (str)
        S (int): Grid size
        B (int): Number of bounding boxes per cell
        C (int): Number of classes
    """
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]

        # Convert model output and labels from cell format (if "cells") to 
        # a more universal set of bounding boxes
        if pred_format == "cells":
            true_bboxes = cellboxes_to_boxes(labels, S=S, B=B, C=C)
            bboxes = cellboxes_to_boxes(predictions, S=S, B=B, C=C)
        else:
            # Otherwise assume you have bounding boxes directly
            # in correct format: [class, conf, x, y, w, h]
            # (Modify as needed)
            true_bboxes = labels
            bboxes = predictions

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # Optionally visualize:
            # if batch_idx == 0 and idx == 0:
            #     plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes, box_format=box_format)

            for nms_box in nms_boxes:
                # Prepend training index for MAP usage
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # Many could be background or have low conf => filter
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=14, B=2, C=2):
    """
    Converts raw model predictions in cell format into bounding boxes
    in "midpoint" format [class, confidence, x_mid, y_mid, width, height],
    choosing the best bounding box among the B predicted per cell.

    predictions: tensor of shape (N, S*S*(C + B*5)) or (N, S, S, C + B*5).
                 The typical shape after final reshape is (N, S, S, C + 5*B).
    S: number of grid cells along each dimension
    B: number of bounding boxes per cell
    C: number of classes
    """

    # Move predictions to CPU if not already
    predictions = predictions.to("cpu")

    N = predictions.shape[0]

    # Reshape to [N, S, S, C + 5*B]
    # In many YOLO implementations, the model might already be shaped this way,
    # but we ensure it here:
    predictions = predictions.reshape(N, S, S, C + B * 5)

    # 1) Split off the class probabilities: [N, S, S, C]
    class_probs = predictions[..., :C]
    # 2) The rest are the bounding box predictions: [N, S, S, 5*B]
    bboxes_preds = predictions[..., C:]  # shape [N, S, S, 5*B]

    # Now reshape that so that for each cell we have shape [B, 5]
    # The layout will be [conf, x, y, w, h] repeated B times
    bboxes_preds = bboxes_preds.reshape(N, S, S, B, 5)

    # 3) We want to pick the bounding box with the highest confidence:
    # bboxes_preds[..., 0] is the confidence
    confs = bboxes_preds[..., 0]  # shape [N, S, S, B]
    best_box_idx = torch.argmax(confs, dim=-1)  # shape [N, S, S]

    # We'll gather the bounding boxes using best_box_idx
    # best_box has shape [N, S, S, 5]
    best_boxes = []
    for n in range(N):
        # advanced indexing for each image
        rows, cols = torch.meshgrid(
            torch.arange(S), torch.arange(S), indexing="ij"
        )
        b_idxs = best_box_idx[n, rows, cols]
        best_boxes_n = bboxes_preds[n, rows, cols, b_idxs, :]
        best_boxes.append(best_boxes_n)
    best_boxes = torch.stack(best_boxes, dim=0)  # [N, S, S, 5]

    # best_boxes[..., 0] = confidence
    # best_boxes[..., 1] = x
    # best_boxes[..., 2] = y
    # best_boxes[..., 3] = w
    # best_boxes[..., 4] = h

    # 4) Convert x, y from cell-space to the image-space
    cell_indices = torch.arange(S).repeat(N, S, 1)  # shape [N, S, S]
    # x = (x + cell_column_index) / S
    # y = (y + cell_row_index) / S
    x = (best_boxes[..., 1] + cell_indices) / S  # cell_indices is along dim=2 for x
    y = (best_boxes[..., 2] + cell_indices.permute(0,2,1)) / S
    w = best_boxes[..., 3] / S
    h = best_boxes[..., 4] / S

    # We also get the predicted class
    predicted_class = class_probs.argmax(-1)  # shape [N, S, S]
    # And the best confidence is best_boxes[..., 0]
    best_confidence = best_boxes[..., 0]

    # Concatenate into final shape: [class, conf, x, y, w, h]
    # We'll do this in a single tensor of shape [N, S, S, 6]
    converted_preds = torch.cat(
        (
            predicted_class.unsqueeze(-1).float(),  # [N, S, S, 1]
            best_confidence.unsqueeze(-1),          # [N, S, S, 1]
            x.unsqueeze(-1),
            y.unsqueeze(-1),
            w.unsqueeze(-1),
            h.unsqueeze(-1),
        ),
        dim=-1,
    )

    return converted_preds


def cellboxes_to_boxes(out, S=14, B=2, C=2):
    """
    Converts the output of convert_cellboxes (for each image) into a Python list
    of bounding boxes [class, confidence, x_mid, y_mid, w, h].
    Returns a list of length batch_size, each element is a list of bboxes for that image.
    """
    # convert_cellboxes => shape [N, S, S, 6]
    converted_pred = convert_cellboxes(out, S=S, B=B, C=C)  # [N, S, S, 6]
    N = converted_pred.shape[0]

    # Reshape to [N, S*S, 6]
    converted_pred = converted_pred.reshape(N, S * S, 6)
    # The first column is class => integer
    converted_pred[..., 0] = converted_pred[..., 0].long()

    all_bboxes = []
    for ex_idx in range(N):
        bboxes = converted_pred[ex_idx].tolist()  # shape [S*S, 6]
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
