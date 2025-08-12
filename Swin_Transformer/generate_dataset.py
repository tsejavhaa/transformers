import os
import cv2
import numpy as np
from PIL import Image

def create_directories(dir_path):
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.join(dir_path, 'low_quality'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'color_noise'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'gray_noise'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'color_jpeg'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'gray_jpeg'), exist_ok=True)
    
