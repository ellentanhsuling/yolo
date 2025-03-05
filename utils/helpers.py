import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def download_model():
    """
    Downloads or loads the YOLOv8 model from ultralytics
    """
    # Using YOLOv8 small
    model = YOLO("yolov8s.pt")
    return model

def get_model_info(model):
    """
    Returns information about the model
    """
    # Get model class names
    class_names = model.names
    
    model_info = {
        'model_type': 'YOLOv8',
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    return model_info
