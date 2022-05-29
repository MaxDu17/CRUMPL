import torch
from torchvision import transforms
import numpy as np

inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if torch.cuda.is_available():
    inception_model.to('cuda')


def loss_on_batch(x_smooth, x_generated_smooth):
    inception_model.train(False)
    x_smooth = preprocess(x_smooth)
    x_generated_smooth = preprocess(x_generated_smooth)

    if torch.cuda.is_available():
        x_smooth = x_smooth.to('cuda')
        x_generated_smooth = x_generated_smooth.to('cuda')

    real_dist_logits = inception_model(x_smooth)
    gen_dist_logits = inception_model(x_generated_smooth)
    real_dist = torch.nn.functional.softmax(real_dist_logits, dim=1)
    gen_dist = torch.nn.functional.softmax(gen_dist_logits, dim=1)
    kl_loss = torch.nn.KLDivLoss(reduction='sum')
    return kl_loss(real_dist.log(), gen_dist)
