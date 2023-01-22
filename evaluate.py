from glob import glob
import os
from class_lib.paint import ImagePainter
from class_lib.train import evaluate_ssim
from settings import *

paths = glob(os.path.join(EVALSET_PATH, '*'))
print(f"\nMean SSIM = {evaluate_ssim(paths, ImagePainter(MODEL_PATH, IMAGE_SIZE))}")