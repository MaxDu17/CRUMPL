# Written by Max Du

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
class PixGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.activations = []

        self.encoder = nn.ModuleList([
            self.encoder_block(self.img_C, 64, bn = False),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256),
            self.encoder_block(256, 512),
            self.encoder_block(512, 512),
            self.encoder_block(512, 512),
            self.encoder_block(512, 512)
        ])

        self.decoder = nn.ModuleList([
            self.decoder_block(512, 512, dropout = True),
            self.decoder_block(1024, 512, dropout=True),
            self.decoder_block(1024, 512, dropout=True),
            self.decoder_block(1024, 256, dropout=False),
            self.decoder_block(512, 128, dropout=False),
            self.decoder_block(256, 64, dropout=False),
        ])

        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size = 4, padding = 1, stride = 2),
            nn.Tanh()
        )

    def encoder_block(self, going_in, out, bn = True):
        if bn:
            return nn.Sequential(
                nn.Conv2d(going_in, out, kernel_size = 4, padding = 1, stride = 2),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(negative_slope = 0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(going_in, out, kernel_size = 4, padding = 1, stride = 2),
                nn.LeakyReLU(negative_slope = 0.2)
            )

    def decoder_block(self, going_in, out, dropout = True):
        if dropout:
            return nn.Sequential(
                nn.ConvTranspose2d(going_in, out, kernel_size = 4, padding = 1, stride = 2),
                nn.BatchNorm2d(out),
                nn.Dropout2d(p = 0.5),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(going_in, out, kernel_size=4, padding=1, stride=2),
                nn.Dropout2d(p=0.5),
                nn.ReLU()
            )

    def forward(self, images):
        self.activations.clear()
        x = images
        for module in self.encoder:
            x = module(x)
            self.activations.append(x)
        self.activations.pop()  # discard the last one because it's the embedding

        for module in self.decoder:
            x = module(x)
            x = torch.cat((x, self.activations.pop()), dim = 1)

        return self.out(x)

class PixDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim

        self.encoder = nn.ModuleList([
            self.encoder_block(self.img_C, 64, bn = False),
            self.encoder_block(64, 128),
            self.encoder_block(128, 256),
            self.encoder_block(256, 512), # 8 x 8, a 16 x 16 receptive field
            # self.encoder_block(512, 512), # 4 x 4, a 32 x 32 receptive field
        ])
        self.out = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 1, padding = 0, stride = 1), # a 1x1 conv
            nn.Sigmoid()
        )

    def encoder_block(self, going_in, out, bn = True):
        if bn:
            return nn.Sequential(
                nn.Conv2d(going_in, out, kernel_size = 4, padding = 1, stride = 2),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(negative_slope = 0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(going_in, out, kernel_size = 4, padding = 1, stride = 2),
                nn.LeakyReLU(negative_slope = 0.2)
            )

    def forward(self, images):
        x = images
        for module in self.encoder:
            x = module(x)
        return self.out(x)


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.activations = []

        self.convs = nn.ModuleList([
            nn.Conv2d(self.img_C, 8, kernel_size=7, padding = 3, stride=1),
            nn.Conv2d(8, 8, kernel_size=7, padding=3, stride=1),
            nn.Conv2d(8, 16, kernel_size=2, stride=2), #pooling
            nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),  # pooling
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),  # pooling
        ])

    def forward(self, images):
        self.activations.clear() #just in case!
        x = images
        for module in self.convs:
            x = module(x)
            x = nn.functional.leaky_relu(x)
            self.activations.append(x)
        self.activations.pop() #discard the last one because it's the embedding
        return x, self.activations


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.activations = []

        #TODO intancenorm
        self.convs = nn.ModuleList([
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # pooling
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # pooling
            nn.ConvTranspose2d(32, 16, kernel_size=5, padding=2, stride=1),
            nn.ConvTranspose2d(32, 16, kernel_size=5, padding=2, stride=1),
            nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2),  # pooling
            nn.ConvTranspose2d(16, 8, kernel_size=7, padding=3, stride=1),
            nn.ConvTranspose2d(16, self.img_C, kernel_size=7, padding = 3, stride=1),
        ])

    def forward(self, encoding, activations):
        x = encoding
        for module in self.convs:
            x = module(x)
            x = nn.functional.leaky_relu(x)
            if len(activations) > 0: #passthrough connections
                x = torch.cat((x, activations.pop()), dim = 1)
                # x = x + activations.pop()
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.activations = []

        self.convs = nn.ModuleList([
            nn.Conv2d(self.img_C * 2, 8, kernel_size=7, padding = 3, stride=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=7, padding=3, stride=1),
            nn.Conv2d(8, 16, kernel_size=2, stride=2), #pooling
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(16, 16, kernel_size=2, stride=2),  # pooling
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(16, 16, kernel_size=2, stride=2),  # pooling
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(16, 16, kernel_size=2, stride=2),  # pooling
        ])

        self.classifier = nn.ModuleList([
            nn.Linear(1024, 1)
        ])

    def forward(self, a, b):
        self.activations.clear() #just in case!
        x = torch.cat([a, b], dim = 1)
        for module in self.convs:
            x = module(x)
            x = nn.functional.leaky_relu(x)
            # nn.functional.leaky_relu(module(x))
            # x = torch.relu(module(x))
        # x = self.convs(x)
        x = torch.flatten(x, start_dim = 1) # so you don't flatten the batch
        # print(x)
        for module in self.classifier:
            x = module(x)
        return x
