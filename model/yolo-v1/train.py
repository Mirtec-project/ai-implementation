import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from model import Yolov1
from dataset import VOCDataset, LabelmeDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from torch.utils.data import DataLoader
from loss import YoloLoss

seed = 123

torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "checkpoint_epoch_230.pth.tar"
IMAGE_DIR = "dataset/PascalVOC_YOLO/images"
LABEL_DIR = "dataset/PascalVOC_YOLO/labels"

print(DEVICE)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
    num_classes = 2
    s_size = 14
    model = Yolov1(split_size=s_size, num_boxes=2, num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(C=num_classes, S=s_size, B=2)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # train_dataset = VOCDataset(
    #     "dataset/PascalVOC_YOLO/100examples.csv",
    #     transform=transform,
    #     img_dir=IMAGE_DIR,
    #     label_dir=LABEL_DIR,
    # )

    train_dataset = LabelmeDataset(
        json_dir="dataset/2차GT-2class",
        img_dir="dataset/2차GT-2class",
        transform=transform,
        S=s_size,
        B=2,
        C=num_classes,
    )

    test_dataset = LabelmeDataset(
        json_dir="dataset/데이터 전달 3 - 3차 GT 작업",
        img_dir="dataset/데이터 전달 3 - 3차 GT 작업",
        transform=transform,
        S=s_size,
        B=2,
        C=num_classes,
    )


    # test_dataset = VOCDataset(
    #     "dataset/PascalVOC_YOLO/test.csv",
    #     transform=transform,
    #     img_dir=IMAGE_DIR,
    #     label_dir=LABEL_DIR,
    # )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):

        for x, y in test_loader:
           x = x.to(DEVICE)
           for idx in range(8):
               bboxes = cellboxes_to_boxes(model(x))
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

           import sys
           sys.exit()
        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, 
            model, 
            iou_threshold=0.5, 
            threshold=0.4, 
            device=DEVICE,
            S=s_size,
            B=2,
            C=num_classes,
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint",
            num_classes=num_classes
        )
        print(f"Train mAP: {mean_avg_prec}")

        # mAP가 일정 기준(예:0.9) 이상이면 기존 파일명으로 저장
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

        # 10에포크마다 다른 이름으로 체크포인트 저장
        if epoch % 10 == 0 and epoch != 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch}.pth.tar")

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()

