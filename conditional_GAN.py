import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchsummary import  summary
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torch.cuda import amp
import torch.nn.functional as F
import torch
import torchvision.utils as vutils
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

NUM_CLASSES_GENERATE = 27
IMAGE_SIZE = 64
NUM_CHANNELS = 3
NOISE_SIZE = 150
FEATURE_MAP_GEN = 64

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES_GENERATE, NUM_CLASSES_GENERATE)
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.ConvTranspose2d(NOISE_SIZE + NUM_CLASSES_GENERATE, FEATURE_MAP_GEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_GEN * 8),
            nn.ReLU(True),
            # state size. (FEATURE_MAP_GEN*8) x 4 x 4
            nn.ConvTranspose2d(FEATURE_MAP_GEN * 8, FEATURE_MAP_GEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_GEN * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( FEATURE_MAP_GEN * 4, FEATURE_MAP_GEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_GEN * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( FEATURE_MAP_GEN * 2, FEATURE_MAP_GEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_GEN),
            nn.ReLU(True),

            nn.ConvTranspose2d( FEATURE_MAP_GEN, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()

        )
    def forward(self, noise_input, labels):
       
        gen_input = torch.cat((self.label_emb(labels).unsqueeze(2).unsqueeze(3), noise_input), 1)

        image = self.main(gen_input)

        image = image.view(image.size(0), *(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        return image
