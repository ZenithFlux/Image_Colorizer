import torch
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from .nn import MainModel, DEVICE
from .utility import AverageCalculator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .paint import ImagePainter
    from torch.utils.data import DataLoader
    from numpy import ndarray

def pretrain(model: MainModel, train_dl: 'DataLoader', lr: float,  epochs: int, save_path: str):    
    optimizer = torch.optim.Adam(model.net_g.parameters(), lr)
    for e in range(epochs):
        print(f'\nGenerator Pretraining Epoch {e+1}/{epochs}:')
        avg_calc = AverageCalculator()
        
        for data in tqdm(train_dl):
            L, ab = data['L'].to(DEVICE), data['ab'].to(DEVICE)
            preds = model.net_g(L)
            loss = model.L1criterion(preds, ab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            avg_calc.update(loss.item())
        
        avg_loss = avg_calc.average()    
        print(f'Epoch {e+1}: Avg. Loss = {avg_loss}')
        
        torch.save(model.state_dict(), save_path)
        print("Model saved")
    
    print("\nGenerator Pretrained!")

def train_g(model: MainModel, train_dl: 'DataLoader', epochs: int, save_path: str):
    for epoch in range(epochs):
        print(f'\nGenerator Training Epoch {epoch+1}/{epochs}:')
        avg_calc_gan = AverageCalculator()
        avg_calc_L1 = AverageCalculator()
        
        model.set_requires_grad(model.net_d, False)
        model.net_g.train()
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.forward()
            model.opt_g.zero_grad()
            model.backward_g()
            model.opt_g.step()
            
            avg_calc_gan.update(model.gan_loss_g.item())
            avg_calc_L1.update(model.L1_loss_g.item())
          
        avg_gan, avg_L1 = avg_calc_gan.average(), avg_calc_L1.average()
        
        print(f'GAN Loss = {avg_gan:.3f}, L1 Loss = {avg_L1:.3f}')
        
        torch.save(model.state_dict(), save_path)
        print("Model saved")
    
    print("\nTraining Complete!")
    
def train_d(model: MainModel, train_dl: 'DataLoader', epochs: int, save_path: str):
    for epoch in range(epochs):
        print(f'\nDiscriminator Training Epoch {epoch+1}/{epochs}:')
        avg_calc_d = AverageCalculator()
        
        model.set_requires_grad(model.net_d, True)
        model.net_d.train()
        for data in tqdm(train_dl):
            model.setup_input(data)
            with torch.no_grad(): model.forward()
            model.opt_d.zero_grad()
            model.backward_d()
            model.opt_d.step()
            
            avg_calc_d.update(model.loss_d.item())
          
        avg_d = avg_calc_d.average()
        
        print(f'Discriminator Loss = {avg_d:.3f}')
        
        torch.save(model.state_dict(), save_path)
        print("Model saved")
    
    print("\nTraining Complete!")
    
def train_gan(model: MainModel, train_dl: 'DataLoader', epochs: int, save_path: str):
    for epoch in range(epochs):
        print(f'\nGAN Training Epoch {epoch+1}/{epochs}:')
        avg_calc_d = AverageCalculator()
        avg_calc_gan = AverageCalculator()
        avg_calc_L1 = AverageCalculator()

        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            
            avg_calc_d.update(model.loss_d.item())
            avg_calc_gan.update(model.gan_loss_g.item())
            avg_calc_L1.update(model.L1_loss_g.item())
            
        avg_d = avg_calc_d.average()
        avg_gan, avg_L1 = avg_calc_gan.average(), avg_calc_L1.average()
        
        print(f'Discriminator Loss = {avg_d:.3f}')
        print(f'Generator: GAN Loss = {avg_gan:.3f}, L1_Loss = {avg_L1:.3f}')
        
        torch.save(model.state_dict(), save_path)
        print("Model saved")
    
    print("\nTraining Complete!")
    
def evaluate_ssim(img_paths: 'ndarray | list[str]', painter: 'ImagePainter') -> float:
    calc = AverageCalculator()
    for path in tqdm(img_paths):
        img1 = Image.open(path).convert('RGB')
        img1 = img1.resize(painter.image_size, Image.Resampling.BICUBIC)
        img2 = painter.paint(img1)
        img1, img2 = np.array(img1), np.array(img2)
        calc.update(ssim(img1, img2, data_range=255, channel_axis=2))
        
    return calc.average()
        