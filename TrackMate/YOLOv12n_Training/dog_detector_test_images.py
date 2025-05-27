from ultralytics import YOLO
import torch
import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())  # sanity check
    # 200 Epochs - early stopped at 180
    model = YOLO(r'C:\projects\CNNDLAT3\Final\best.pt')

    # Send model to GPU explicitly (optional, usually not needed)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = model.val(data=r'C:\projects\CNNDLAT3\Final\detection\dataset.yaml', split='train', device=0)  # 0 = first GPU
    print(f"Precision:  {metrics.box.mp.mean():.4f}")
    print(f"Recall:     {metrics.box.mr.mean():.4f}")
    print(f"mAP@0.5:    {metrics.box.ap50.mean():.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map.mean():.4f}")

