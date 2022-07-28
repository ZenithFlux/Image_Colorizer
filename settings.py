"""
IMAGE_SIZE-
Set image resolution to train the model on.
It may not be same as the actual resolution of the images.
All images will be scaled to this size before being fed into model.
Output of the model will be of this resolution.
"""
IMAGE_SIZE = (256, 256)

# set the path to folder where all the images for training are stored
DATASET_PATH = "dataset\\coco"

# Number of images to train on
TRAINING_SET_SIZE = 8000

# Try reducing these values if your pc is going out of memory
# Increase these values if your pc is more capable
BATCH_SIZE = 4
NUM_WORKERS = 1

# path for both storing and loading model
MODEL_PATH = "main_model.pt"

PRETRAINING_EPOCHS = 20
TRAINING_EPOCHS = 20

"""
If you want to train completely new image colouring model, make sure to delete 'main_model.pt' file
or else training algorithm will continue training the existing model.
"""