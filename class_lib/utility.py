import torch
import cv2
from PIL import Image
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image
    from numpy import ndarray

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageCalculator:
    def __init__(self):
        self.reset()
        
    def update(self, num):
        self.count += 1
        self.sum += num
        
    def average(self):
        return self.sum/self.count
        
    def reset(self):
        self.sum = self.count = 0.0

def print_gpu_status():
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name())
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated()/1024**3,1), 'GB')
        print('Reserved:   ', round(torch.cuda.memory_reserved()/1024**3,1), 'GB')
        
def cv2_to_pil(img: 'ndarray') -> 'Image':
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    return pil_img

def pil_to_cv2(img: 'Image') -> 'ndarray':
    img = np.array(img)
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2_img