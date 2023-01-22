import os
import torch
from class_lib.nn import MainModel, DEVICE
from class_lib.train import train_gan, pretrain, train_d
from class_lib.data import Datapaths, make_dataloader
from settings import *

def main():
    model = MainModel(image_size=IMAGE_SIZE)

    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)
        print("main_model.pt created\n")
    else:
        model.load_state_dict(torch.load(MODEL_PATH, DEVICE))
        print("main_model.pt loaded\n")
        
    print("Please select:\n1. [Full Training]\n2. Pretrain Generator")
    print("3. Pretrain Discriminator\n4. GAN Training")
    option = int(input("Select no.: "))
    
    if option==1:
        print("\nFull Model Training details:")
        print(f"\tStep 1: Generator Pretraining ({G_PRETRAINING_EPOCHS} epochs)")
        print(f"\tStep 2: Discriminator Pretraining ({D_PRETRAINING_EPOCHS} epochs)")
        print(f"\tStep 3: GAN Training ({GAN_TRAINING_EPOCHS} epochs)")
    
    if option==2 or option==1:
        paths = Datapaths(DATASET_PATH, PRETRAINING_SET_SIZE, 0, 1)
        train_dl = make_dataloader(paths.train_paths, 'train', IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)
        print("\nStarting Generator pretraining...")
        pretrain(model, train_dl, 1e-4, G_PRETRAINING_EPOCHS, MODEL_PATH)
    
    if option==3 or option==1:
        paths = Datapaths(DATASET_PATH, PRETRAINING_SET_SIZE, 0, 1)
        train_dl = make_dataloader(paths.train_paths, 'train', IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)
        print("\nStarting Discriminator pretraining...")
        train_d(model, train_dl, D_PRETRAINING_EPOCHS, MODEL_PATH)
    
    if option==4 or option==1:
        paths = Datapaths(DATASET_PATH, GAN_TRAINING_SET_SIZE, 0, 1)
        train_dl = make_dataloader(paths.train_paths, 'train', IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)
        print("\nStarting GAN training...")
        train_gan(model, train_dl, GAN_TRAINING_EPOCHS, MODEL_PATH)      
    
if __name__ == '__main__':
    main()