from class_lib.paint import ImagePainter, paint_video
from settings import MODEL_PATH, IMAGE_SIZE

def colour_video(video_path: str, save_path: str):
    painter = ImagePainter(MODEL_PATH, IMAGE_SIZE)
    paint_video(video_path, save_path, painter)

if __name__ == "__main__":
    print("This program will colour the given video using deep learning algorithm.")
    print("If input video is coloured than it will be converted to grayscale and then painted.\n")
    
    video_path = input("Enter video path: ")
    save_path = input("Enter colourized video save path: ")
    
    colour_video(video_path, save_path)