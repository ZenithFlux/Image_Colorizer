# Movie_Colorizer

This program can colour B&W movies using deep learning.

Some example images painted by this program:

<img src="https://i.ibb.co/NLXP0ZQ/gray.jpg" alt="Black & White Image 1" width="20%"/> &nbsp;&nbsp;&nbsp;<img src="https://i.ibb.co/S35NVN0/coloured.jpg" alt="Coloured Image 1" width="20%"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.ibb.co/VmwcQxB/gray2.jpg" alt="Black & White Image 2" width="20%"/> &nbsp;&nbsp;&nbsp;<img src="https://i.ibb.co/SX1Wgvw/coloured2.jpg" alt="Coloured Image 2" width="20%"/>

<br/>

<img src="https://i.ibb.co/LdxHxjd/gray3.jpg" alt="Black & White Image 3" width="20%"/> &nbsp;&nbsp;&nbsp;<img src="https://i.ibb.co/QNCBhjp/coloured3.jpg" alt="Coloured Image 3" width="20%"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://i.ibb.co/jT87x1K/gray4.jpg" alt="Black & White Image 4" width="20%"/> &nbsp;&nbsp;&nbsp;<img src="https://i.ibb.co/p4Wh5bn/coloured4.jpg" alt="Coloured Image 4" width="20%"/>

## Installing Dependencies

Type this in command line to install all dependencies except PyTorch:

```pip install -r requirements.txt```

***Note:-*** [Install PyTorch](https://pytorch.org/get-started/locally/) according to your specific pc specs.

## Dataset

Dataset I used for training: [COCO](https://cocodataset.org/#download) 2017 Unlabeled images (After filtering out grayscale images)

Model trained on this dataset for 256x256 images- [Download from G-Drive](https://drive.google.com/file/d/1ejIH8i-jUlci_v9p71nGfp9A76SvtzgJ/view?usp=sharing)

## How to Use

*Note (For Non-Technical Users) -* Users who have no knowledge of deep learning or pytorch are advised to only edit values in ***'settings.py'***. Most of the training and application usage can be handled using this file.

**train_model.py -** You can train your own image colouring model by running this script. All training settings can be adjusted from 'settings.py'.

**colour_images.py -** Run this file to colour your grayscale images using a trained model. A trained model should be on location MODEL_PATH mentioned in 'settings.py'.

**colour_video.py -** Run this file to colour your grayscale videos using a trained model. A trained model should be on location MODEL_PATH mentioned in 'settings.py'.

# Credits

Heavily inspired from [Colorizing black & white images with U-Net and conditional GAN](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)