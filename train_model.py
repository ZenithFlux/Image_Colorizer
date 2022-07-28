import os
import torch
from class_lib.nn import MainModel, unet_resnet18, train_model, pretrain, DEVICE
from class_lib.data import Datapaths, make_dataloader
from settings import *

def main():

    paths = Datapaths(DATASET_PATH, TRAINING_SET_SIZE, 0)
    train_dl = make_dataloader(paths.train_paths, 'train', IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)

    if not os.path.exists(MODEL_PATH):
        net_g = unet_resnet18(1, 2, IMAGE_SIZE)
        opt = torch.optim.Adam(net_g.parameters(), 1e-4)
        loss_func = torch.nn.L1Loss()
        print("\nStarting unet pretraining...")
        pretrain(net_g, train_dl, opt, loss_func, PRETRAINING_EPOCHS)
            
        model = MainModel(net_g)
        torch.save(model.state_dict(), MODEL_PATH)
        print('\nmain_model.pt saved')
        print("\nStarting MainModel training...")
        train_model(model, train_dl, TRAINING_EPOCHS, MODEL_PATH)
        
    else:
        model = MainModel(image_size = IMAGE_SIZE)
        model.load_state_dict(torch.load(MODEL_PATH, DEVICE))
        print("\nmain_model.pt loaded")
        print("Continuing MainModel training...")
        train_model(model, train_dl, TRAINING_EPOCHS, MODEL_PATH)      
    
    
    input("Press Enter to continue...")
    
if __name__ == '__main__':
    main()