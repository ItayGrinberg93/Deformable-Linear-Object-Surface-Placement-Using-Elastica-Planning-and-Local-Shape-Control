from ultralytics import YOLO
from ultralytics import SAM
import os
import numpy as np
import cv2 as cv
import sys
import time
import torch
import torch.nn as nn

def Train():
    # check for GPU
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    # Making the code device-agnostic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data load
    data_path = './Path_to_data_folder/data.yaml'

    # Load a model
    model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Transferring the model to a CUDA enabled GPU
    # model = model.to(device)
    # Train the model
    results = model.train(task='segmant', data=data_path, epochs=100, imgsz=640, device=device)

    return results


if __name__ == '__main__':
    results = 0
    # Load a model
    # model = YOLO('../scripts/runs/segment/train10/weights/best.pt')  # build a new model from YAML
    yolo_model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
    # model = SAM('sam_b.pt') # load a pretrained model.
    # model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Model train
    model = Train()

