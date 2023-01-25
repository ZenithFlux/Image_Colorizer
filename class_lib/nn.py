from torch import nn, optim
import torch
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from torchvision.models import ResNet18_Weights
from fastai.vision.models.unet import DynamicUnet
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to create generator
def unet_resnet18(in_channels: int, out_channels: int, image_size: tuple[int, int]) -> DynamicUnet:
    body = create_body(resnet18(weights=ResNet18_Weights.DEFAULT), 
                       pretrained=True, n_in=in_channels, cut=-2)
    model = DynamicUnet(body, out_channels, image_size).to(DEVICE)
    return model

class GANLoss(nn.Module):
    def __init__(self, real_label: float=1.0, fake_label: float=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, pred: 'Tensor', is_real: bool) -> 'Tensor':
        if is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
            
        return labels.expand_as(pred)
    
    def __call__(self, pred: 'Tensor', is_real: bool) -> 'Tensor':
        labels = self.get_labels(pred, is_real)
        return self.loss(pred, labels)
        
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, n_filters: int=64, hidden_layers: int=3):
        super().__init__()
        model = [self.get_layers(in_channels, n_filters, norm=False)]
        model += [self.get_layers(n_filters * 2**i, n_filters * 2**(i+1),
                                  s=1 if i == (hidden_layers-1) else 2) for i in range(hidden_layers)]
        model += [self.get_layers(n_filters * 2**hidden_layers, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)
    
    def get_layers(self, in_c: int, out_c: int, k: int=4, s: int=2, p: int=1, norm: bool=True, act: bool=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(out_c)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, X: 'Tensor') -> 'Tensor':
        return self.model(X)
  
class MainModel(nn.Module):
    def __init__(self, net_g: DynamicUnet | None = None, image_size: tuple[int, int] | None = None,
                 lr_g: float=2e-4, lr_d: float=2e-4, beta1: float=0.5, beta2: float=0.999, lambda_L1: float=100.):
        super().__init__()
        self.lambda_L1 = lambda_L1
        
        if net_g is not None:
            self.net_g = net_g.to(DEVICE)
        elif image_size is not None:
            self.net_g = unet_resnet18(1, 2, image_size).to(DEVICE)
        else:
            raise ValueError("A value must be passed for either 'net_g' or 'image_size' in MainModel().")
            
        self.net_d = init_model(PatchDiscriminator(3))
        self.GANcriterion = GANLoss().to(DEVICE)
        self.L1criterion = nn.L1Loss()
        self.opt_g = optim.Adam(self.net_g.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.opt_d = optim.Adam(self.net_d.parameters(), lr=lr_d, betas=(beta1, beta2))

    def setup_input(self, data: 'dict[str, Tensor]'):
        self.L = data['L'].to(DEVICE)
        self.ab = data['ab'].to(DEVICE)
        
    def forward(self, L: 'Tensor | None' = None) -> 'Tensor | None':
        if L is not None:
            L = L.to(DEVICE)
            self.net_g.eval()
            with torch.no_grad(): ab = self.net_g(L)
            self.net_g.train()
            return ab
            
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
        
    def set_requires_grad(self, model: 'Module', requires_grad: bool=True):
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

# This function initializes weights of a model
def init_model(model: 'Module', gain: float = 0.02) -> 'Module':
    model = model.to(DEVICE)
    
    def init_weights(layer):
        classname = layer.__class__.__name__
        with torch.no_grad():
            if hasattr(layer, 'weight') and 'Conv' in classname:
                nn.init.normal_(layer.weight, 0.0, gain)
            
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(layer.weight, 1., gain)
                nn.init.constant_(layer.bias, 0.)
                
    model.apply(init_weights)
    return model