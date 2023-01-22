"""
IMAGE_SIZE-
Set image resolution to train the model on.
It may not be same as the actual resolution of the images.
All images will be scaled to this size before being fed into model.
Output of the model will be of this resolution.
"""
IMAGE_SIZE = (256, 256)

# path for both saving and loading model, should end with '.pt'
MODEL_PATH = "main_model.pt"

"""
If you want to train completely new image colouring model, 
make sure that there is no '.pt' file with same name as MODEL_PATH 
or else training algorithm will continue training the existing model.
"""


#---------------------------Training Settings--------------------------------

# set the path to folder where all the images for training are stored
DATASET_PATH = "dataset\\coco"
# dataset path for SSIM evaluation
EVALSET_PATH = "dataset\\val"

# Number of images to train on
PRETRAINING_SET_SIZE = 100000
GAN_TRAINING_SET_SIZE = 100000

# Try reducing these values if your pc is going out of memory
# Increase these values if your pc is more capable
BATCH_SIZE = 4
NUM_WORKERS = 1

G_PRETRAINING_EPOCHS = 10
D_PRETRAINING_EPOCHS = 4
GAN_TRAINING_EPOCHS = 10