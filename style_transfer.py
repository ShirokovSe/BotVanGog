import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import os
import numpy as np


def find_mean_std(input, eps = 1e-5):
  batch_size, channels, height, weight = input.size()
  input_std = torch.sqrt(input.view(batch_size, channels,-1).var(dim=2) + eps).view(batch_size, channels,1,1)
  input_mean = torch.mean(input.view(batch_size, channels,-1), dim = 2).view(batch_size, channels,1,1)
  
  return input_mean, input_std


def AdaIN(content, style):
  content_mean, content_std = find_mean_std(content)
  style_mean, style_std = find_mean_std(style)

  return style_std * ((content - content_mean) / content_std ) + style_mean


class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.model = nn.Sequential(

        nn.Conv2d(512,256, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Upsample(scale_factor = 2, mode = 'nearest'),

        nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Conv2d(256,128, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor = 2,mode = 'nearest'),
 
        nn.Conv2d(128,128, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Conv2d(128,64,kernel_size=3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Upsample(scale_factor = 2, mode='nearest'),
        nn.Conv2d(64,64, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect'),
        nn.ReLU(inplace = True),
        nn.Conv2d(64,3, kernel_size = 3, padding = 1, padding_mode='reflect'),  
    )

  def forward(self,x):
    return self.model(x)

class Net(nn.Module):
  def __init__(self,decoder):
    super().__init__()
    self.encoder = torchvision.models.vgg19(pretrained=True).features[:21]
    self.decoder = decoder
    self.mse_loss = nn.MSELoss()


    for module in self.encoder.modules():
        classname = module.__class__.__name__
        if 'Conv' in classname:
            module.padding_mode = 'reflect'

    for parameter in self.encoder.parameters():
      parameter.requires_grad_(False)

  def decode(self,x):
    return self.decoder(x)

  def encode(self,x):
    return self.encoder(x)

  def encode_per_layer(self,x):

    features = []

    for layer_num,layer in enumerate(self.encoder):
      x = layer(x)

      if layer_num in [1,6,11,21]:
        features.append(x)

    return features

  def content_loss(self,x, content):
    return self.mse_loss(x, content)


  def style_loss(self, x, style):
    mean_st, std_st = find_mean_std(style)
    mean_inp, std_inp = find_mean_std(x)

    return self.mse_loss(mean_inp, mean_st) + self.mse_loss(std_inp, std_st)

  def forward(self, content, style, alpha = 1.0):
    style_f = self.encode_per_layer(style)
    content_f = self.encode_per_layer(content)

    normal = AdaIN(content_f[-1],style_f[-1])

    generated = self.decoder((1 - alpha) * content_f[-1] + alpha * normal)
    generated = self.encode_per_layer(generated[-1])

    loss_cont = self.content_loss(generated[-1], normal)
    loss_style = self.style_loss(generated[0], style_f[0])
    
    for layer in range(1,4):
        loss_style += self.style_loss(generated[layer], style_f[layer])

    return loss_cont, loss_style
