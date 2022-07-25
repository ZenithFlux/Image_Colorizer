from PIL import Image
from skimage.color import lab2rgb
from torchvision import transforms
from torch import cat, device, load
import numpy as np
from .nn import MainModel

class ImagePainter:
    def __init__(self, model_path: str, image_size: tuple[int, int]):
        self.model = MainModel()
        self.model.load_state_dict(load(model_path))
        self.image_size = image_size
    
    def get_L(self, img):
        img = img.convert('L')
        img = transforms.Resize(self.image_size, transforms.InterpolationMode.BICUBIC)(img)
        img = (transforms.ToTensor()(img) * 100) / 50.0 - 1       # Brings L between -1 and 1
        return img
    
    def lab_to_image(self, L, ab):
        L = (L + 1) * 50.0
        ab = ab * 128
        Lab = cat([L, ab], dim=1).permute(0, 2, 3, 1)[0].cpu().numpy()
        img_arr = lab2rgb(Lab) * 255
        img_arr = img_arr.astype(np.uint8)
        img = Image.fromarray(img_arr)
        return img
    
    def paint(self, image: Image.Image) -> Image.Image:
        """Takes PIL image as input, returns PIL image as output."""
        L = self.get_L(image)
        L = L.unsqueeze(0)
        ab = self.model(L)
        ab = ab.to(device('cpu'))
        img = self.lab_to_image(L, ab)
        
        return img
        
        
        