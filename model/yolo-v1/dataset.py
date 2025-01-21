"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


class LabelmeDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None, S=7, B=2, C=2):
        """
        json_dir: labelme json 파일들이 있는 디렉토리
        img_dir: 이미지 파일들이 있는 디렉토리
        transform: 이미지 변환을 위한 transform
        S: 그리드 크기
        B: 박스 개수
        C: 클래스 개수 (CMounting, CSolder)
        """
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
        # 클래스 매핑
        self.classes = {
            "CMounting": 0,
            "CSolder": 1
        }
        
        # json 파일 리스트 생성
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, index):
        json_path = os.path.join(self.json_dir, self.json_files[index])
        
        # json 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 이미지 로드
        img_path = os.path.join(self.img_dir, data['imagePath'])
        image = Image.open(img_path).convert("RGB")
        
        # 박스 정보 추출
        boxes = []
        for shape in data['shapes']:
            label = shape['label']
            if label not in self.classes:
                continue
                
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 좌표 정규화 (0~1 사이 값으로)
            x1, x2 = min(x1, x2) / data['imageWidth'], max(x1, x2) / data['imageWidth']
            y1, y2 = min(y1, y2) / data['imageHeight'], max(y1, y2) / data['imageHeight']
            
            # [class_label, x_center, y_center, width, height] 형식으로 변환
            class_label = self.classes[label]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            boxes.append([class_label, x_center, y_center, width, height])
            
        boxes = torch.tensor(boxes)
        
        # transform 적용
        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        # 라벨 매트릭스 생성 (VOCDataset과 동일한 형식)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            # i,j는 셀의 행과 열 인덱스
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            # 셀 기준 width, height
            width_cell, height_cell = width * self.S, height * self.S
            
            # 해당 셀에 객체가 없는 경우에만 추가
            if label_matrix[i, j, self.C] == 0:
                # 객체 존재 여부 설정
                label_matrix[i, j, self.C] = 1
                
                # 박스 좌표 설정
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates
                
                # 클래스 one-hot 인코딩
                label_matrix[i, j, class_label] = 1
                
        return image, label_matrix

# 사용 예시
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    dataset = LabelmeDataset(
        json_dir="dataset/2차GT-2class",
        img_dir="dataset/2차GT-2class",
        transform=None  # 필요한 transform 추가
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # 데이터 로드 테스트
    for images, labels in dataloader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break