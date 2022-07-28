# Movie_Colorizer

This program colors B&W movies using deep learning.

Type this in command line to install all dependencies except PyTorch:

```pip install -r requirements.txt```

***Note:-*** [Install PyTorch](https://pytorch.org/get-started/locally/) according to your specific pc specs.

## Dataset

Dataset I used for training: [COCO](https://cocodataset.org/#download) 2017 Unlabeled images (After filtering out grayscale images)

Model trained on this dataset for 256x256 images- [Download from G-Drive](https://drive.google.com/file/d/1ejIH8i-jUlci_v9p71nGfp9A76SvtzgJ/view?usp=sharing)

## How to Use

*Note (For Non-Technical Users) -* Users who have no knowledge of deep learning or pytorch are advised to only edit values in 'settings.py'. Most of the training and application usage can be handled using this file.

**train_model.py -** You can train your own image colouring model by running this script. All training settings can be adjusted from 'settings.py'.

**colour_images.py -** Run this file to colour your grayscale images using a trained model. A trained model should be on location MODEL_PATH mentioned in 'settings.py'.

**colour_video.py -** Run this file to colour your grayscale videos using a trained model. A trained model should be on location MODEL_PATH mentioned in 'settings.py'.

# Credits

Heavily inspired from [Colorizing black & white images with U-Net and conditional GAN](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)