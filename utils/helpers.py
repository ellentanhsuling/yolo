import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def download_model():
    """
    Downloads or loads the YOLOv5 model
    """
    # Using YOLOv5s - a small version of YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Set to evaluation mode
    model.eval()
    return model

def get_model_info(model):
    """
    Returns information about the model
    """
    # Get model class names
    class_names = model.names
    
    # Get model details
    model_info = {
        'model_type': model.model.model_type if hasattr(model.model, 'model_type') else 'YOLOv5',
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    return model_info
