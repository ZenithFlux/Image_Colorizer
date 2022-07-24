"""
IMAGE_SIZE-
Set image resolution to train the model on.
It may not be same as the actual resolution of the images.
All images will be scaled to this size before being fed into model.
Output of the model will be of this resolution.
"""
IMAGE_SIZE = (256, 256)

DATASET_PATH = "dataset\\COCOval2014"
TRAINING_SET_SIZE = 8000

# Try reducing these values if your pc is going out of memory
BATCH_SIZE = 4
NUM_WORKERS = 1

PRETRAINING_EPOCHS = 20
TRAINING_EPOCHS = 20

"""
If you want to train completely new image colouring model, make sure to delete 'main_model.pt' file
or else training algorithm will continue training the existing model.
"""