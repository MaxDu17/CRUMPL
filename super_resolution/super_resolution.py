from matplotlib import pyplot as plt

from utils import convert_image
from PIL import Image
import torch
import numpy as np
import os
import cv2

srgan_checkpoint = "../checkpoint_srgan.pth.tar"
device = 'cpu'

srgan_generator = torch.load(srgan_checkpoint, map_location=lambda storage, loc: storage)['generator'].to(device)
base_directory = '../data/generated_images'


def super_res(path):
    hr_img = Image.open(f'{base_directory}/{path}', mode="r")
    hr_img = hr_img.convert('RGB')

    # lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)), Image.BICUBIC)

    sr_img_srgan = srgan_generator(convert_image(hr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    # combined = torch.cat([hr_img, sr_img_srgan], dim=2)

    sr_np = np.array(sr_img_srgan)

    orig_np = np.array(hr_img)
    orig_np = cv2.resize(orig_np, sr_np.shape[:2], interpolation=cv2.INTER_NEAREST)

    diff = sr_np - orig_np

    # print(sr_np.shape, orig_np.shape)

    combined = np.concatenate([sr_np, orig_np, diff], axis=0)
    plt.imshow(combined)

    plt.show()

i = 0
for img in os.listdir(base_directory):
    super_res(img)
    i += 1
    if i == 10:
        break
