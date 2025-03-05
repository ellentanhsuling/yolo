import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def download_model():
    """
    Downloads a YOLOv5 model directly without AutoShape dependency
    """
    # Load directly from an older, compatible version of YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                          path='https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt')
    model.eval()
    return model

def get_model_info(model):
    """
    Returns information about the model
    """
    # Get model class names (assuming COCO classes if not available)
    try:
        class_names = model.names
    except AttributeError:
        # Fallback to COCO class names
        class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            # ... add more classes as needed
        }
    
    model_info = {
        'model_type': 'YOLOv5',
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    return model_info
