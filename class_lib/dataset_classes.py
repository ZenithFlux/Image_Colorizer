import glob
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.color import rgb2lab

class Datapaths:
    def __init__(self, path: str, train_size: int, validation_size: int, randomizer_seed: int = None):
        paths = glob.glob(os.path.join(path, '*')) # Getting path of every file in the dataset folder
        
        if randomizer_seed is not None:
            np.random.seed(randomizer_seed) # dont randomize if user wants predictive results and enters a seed
            
        total_paths = np.random.choice(paths, train_size + validation_size, replace = False) # randomly choose data
        self.train_paths = total_paths[:train_size]
        self.val_paths = total_paths[train_size:]   #split training and validation datasets
        np.random.seed()    # reset the seed
        
        
    def reshuffle(self, cross_shuffle: bool = False):
        # cross_shuffle means shuffling between training and validation dataset
                
        if cross_shuffle:
            total_paths = np.append(self.train_paths, self.val_paths)
            train_size = self.train_paths.shape[0]
            np.random.shuffle(total_paths)
            self.train_paths = total_paths[:train_size]
            self.val_paths = total_paths[train_size:]
            
        else:
            np.random.shuffle(self.train_paths)
            np.random.shuffle(self.val_paths)
            

class ImagesDataset(Dataset):
    def __init__(self, paths, dataset_type: str = 'train', image_size: tuple = (256, 256)):
        self.paths = paths
        
        BICUBIC = transforms.InterpolationMode.BICUBIC
        
        if dataset_type == 'train':
            self.transformations = transforms.Compose([transforms.Resize(image_size, BICUBIC),
                                                       transforms.RandomHorizontalFlip()])
        else:
            self.transformations = transforms.Resize(image_size, BICUBIC)
        
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        img = self.transformations(img)
            
        img = rgb2lab(img).astype("float32")
        img = transforms.ToTensor()(img)
        L = img[[0], ...]/50.0 - 1      # Bring between -1 ans 1
        ab = img[[1, 2], ...] / 128     # Bring between -1 and 1
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
    

'''
Function to create dataloaders:
- pin_memory is set to True by default bcz my pc has CUDA available
'''
def make_dataloader(paths, batch_size: int = 16, num_workers: int = 2, pin_memory: bool = True, **kwargs):
    dataset = ImagesDataset(paths, **kwargs)
    dataloader = DataLoader(dataset, batch_size= batch_size, num_workers= num_workers, pin_memory= pin_memory)
    return dataloader