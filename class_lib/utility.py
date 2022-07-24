import torch

DEVICE = DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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