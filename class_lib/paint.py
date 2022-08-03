from PIL import Image
from skimage.color import lab2rgb
from torchvision import transforms
from torch import cat, device, load
import cv2
from tqdm import tqdm
import numpy as np
from .nn import MainModel, DEVICE
from .utility import pil_to_cv2, cv2_to_pil

class ImagePainter:
    def __init__(self, model_path: str, image_size: tuple[int, int]):
        self.model = MainModel(image_size=image_size)
        self.model.load_state_dict(load(model_path, DEVICE))
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
        
def paint_video(video_path: str, save_path: str, painter: ImagePainter):
    """Output video will be of same resolution as 'image_size' of ImagePainter object."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Invalid Video Path')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, painter.image_size)
    print(f"\nColouring Video at {fps:.2f}fps and {painter.image_size[0]}x{painter.image_size[1]} resolution:")
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        
        if ret:
            img = cv2_to_pil(frame)
            img = painter.paint(img)
            frame = pil_to_cv2(img)
            
            out.write(frame)
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Coloured video successfully saved\n") 