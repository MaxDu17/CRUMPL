import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms


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
            x = torch.relu(module(x))
            self.activations.append(x)
        self.activations.pop() #discard the last one because it's the embedding
        return x, self.activations


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.activations = []

        #TODO batchnorm
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
            x = torch.relu(x)
            if len(activations) > 0: #passthrough connections
                x = torch.concat((x, activations.pop()), dim = 1)
                # x = x + activations.pop()
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = tuple(input_dim)
        self.img_C, self.img_H, self.img_W = self.input_dim
        self.activations = []

        self.convs = nn.ModuleList([
            # nn.Conv2d(self.img_C, 8, kernel_size=7, padding = 3, stride=1),
            # nn.Conv2d(8, 8, kernel_size=7, padding=3, stride=1),
            # nn.Conv2d(8, 16, kernel_size=2, stride=2), #pooling
            # nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1),
            # nn.Conv2d(16, 16, kernel_size=5, padding=2, stride=1),
            # nn.Conv2d(16, 32, kernel_size=2, stride=2),  # pooling
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            # nn.Conv2d(32, 64, kernel_size=2, stride=2),  # pooling
        ])
        self.resizing = transforms.Resize((224))

        resnet = models.resnet18(pretrained=False)
        resnet.fc = nn.Identity()
        self.convs = resnet

        self.classifier = nn.ModuleList([
            nn.Linear(512, 1)
        ])

    def forward(self, images):
        self.activations.clear() #just in case!
        x = images
        x = self.resizing(x)
        x = self.convs(x)
        x = torch.flatten(x, start_dim = 1) # so you don't flatten the batch
        # print(x)
        for module in self.classifier:
            x = module(x)
        return x
