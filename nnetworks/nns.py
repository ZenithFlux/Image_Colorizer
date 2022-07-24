from torch import nn, optim
import torch
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from tqdm import tqdm
from class_lib.utility import AverageCalculator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to create generator
def unet_resnet18(in_channels: int, out_channels: int, size: tuple[int, int] = (256, 256)) -> DynamicUnet:
    body = create_body(resnet18(True), pretrained=True, n_in=in_channels, cut=-2)
    model = DynamicUnet(body, out_channels, size).to(DEVICE)
    return model  

class GANLoss(nn.Module):
    def __init__(self, real_label: float=1.0, fake_label: float=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, pred, is_real: bool):
        if is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
            
        return labels.expand_as(pred)
    
    def __call__(self, pred, is_real: bool):
        labels = self.get_labels(pred, is_real)
        return self.loss(pred, labels)
        
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, n_filters: int=64, hidden_layers: int=3):
        super().__init__()
        model = [self.get_layers(in_channels, n_filters, norm=False)]
        model += [self.get_layers(n_filters * 2**i, n_filters * 2**(i+1), s=1 if i == (hidden_layers-1) else 2) for i in range(hidden_layers)]
        model += [self.get_layers(n_filters * 2**hidden_layers, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)
    
    def get_layers(self, in_c, out_c, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(out_c)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, X):
        return self.model(X)
  
class MainModel(nn.Module):
    def __init__(self, net_g = unet_resnet18(1, 2), lr_g: float=2e-4, lr_d: float=2e-4, beta1: float=0.5, beta2: float=0.999, lambda_L1: int=100):
        super().__init__()
        self.lambda_L1 = lambda_L1
        self.net_g = net_g.to(DEVICE)
        self.net_d = init_model(PatchDiscriminator(3))
        self.GANcriterion = GANLoss().to(DEVICE)
        self.L1criterion = nn.L1Loss()
        self.opt_g = optim.Adam(self.net_g.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.opt_d = optim.Adam(self.net_d.parameters(), lr=lr_d, betas=(beta1, beta2))

    def setup_input(self, data):
        self.L = data['L'].to(DEVICE)
        self.ab = data['ab'].to(DEVICE)
        
    def forward(self):
        self.fake_ab = self.net_g(self.L)
        
    def backward_d(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        fake_preds = self.net_d(fake_image.detach())
        self.fake_loss_d = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_d(real_image)
        self.real_loss_d = self.GANcriterion(real_preds, True)
        self.loss_d = (self.fake_loss_d + self.real_loss_d) * 0.5
        self.loss_d.backward()
        
    def backward_g(self):
        fake_image = torch.cat([self.L, self.fake_ab], dim=1)
        fake_preds = self.net_d(fake_image)
        self.gan_loss_g = self.GANcriterion(fake_preds, True)
        self.L1_loss_g = self.L1criterion(self.fake_ab, self.ab) * self.lambda_L1
        self.loss_g = self.gan_loss_g + self.L1_loss_g
        self.loss_g.backward()
        
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def optimize(self):
        self.forward()
        self.net_d.train()
        self.set_requires_grad(self.net_d, True)
        self.opt_d.zero_grad()
        self.backward_d()
        self.opt_d.step()
        
        self.net_g.train()
        self.set_requires_grad(self.net_d, False)
        self.opt_g.zero_grad()
        self.backward_g()
        self.opt_g.step()

# This function initialises weights of a model
def init_model(model, gain: float = 0.02):
    model = model.to(DEVICE)
    
    def init_weights(layer):
        with torch.no_grad():
            if hasattr(layer, 'weight') and isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0.0, gain)
            
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight, 1., gain)
                nn.init.constant_(layer.bias, 0.)
                
    model.apply(init_weights)
    return model

def pretrain(model, train_dl, optimizer, loss_func, epochs: int):
    for e in range(epochs):
        print(f'\nPretraining Epoch {e+1}/{epochs}:')
        avg_calc = AverageCalculator()
        
        for data in tqdm(train_dl):
            L, ab = data['L'].to(DEVICE), data['ab'].to(DEVICE)
            preds = model(L)
            loss = loss_func(preds, ab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            avg_calc.update(loss.item())
        
        avg_loss = avg_calc.average()    
        print(f'Epoch {e+1}: Avg. Loss = {avg_loss}')
    
    print("\nGenerator Pretrained!")
    
def train_model(model, train_dl, epochs: int, save_path: str):
    for epoch in range(epochs):
        print(f'\nTraining Epoch {epoch+1}/{epochs}:')
        avg_calc = AverageCalculator()
        
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            
            avg_calc.update(model.loss_g.item())
        
        avg_loss = avg_calc.average()
        print(f'Epoch {epoch+1}: Avg. Loss = {avg_loss}')
        
        torch.save(model.state_dict(), save_path)
        print("Model saved")
    
    print("\nTraining Complete!")