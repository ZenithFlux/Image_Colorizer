from class_lib.paint import ImagePainter
from PIL import Image
from settings import *

def colour_image(input_image_path: str, output_path: str = "", display: bool=True):
    img = Image.open(input_image_path)
    
    painter = ImagePainter(MODEL_PATH, IMAGE_SIZE)
    img = painter.paint(img)
    
    if output_path: img.save(output_path)
    if display: img.show()
    
if __name__ == "__main__":
    print("This program will colour the given image using deep learning algorithm.")
    print("If input image coloured than it will be converted to grayscale and then painted.\n")
    
    input_path = input("Enter image location: ")
    output_path = input("Enter painted image save location (Leave blank to only display image):\n")
    colour_image(input_path, output_path)