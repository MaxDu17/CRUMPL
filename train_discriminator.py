import pickle
import numpy as np
import random
import torch
from torch import nn
import os
import time
from shutil import copyfile
import matplotlib.pyplot as plt
from EncoderDecoder import Discriminator
import csv
from pipeline_whole import CrumpleLibrary
from torch.utils.tensorboard import SummaryWriter
sampler_dataset = pickle.load(open("dataset_10000_small.pkl", "rb"))
sampler_dataset.set_mode("classifier_sample")
print("done loading data")
from utils import *

from torch.utils.data import DataLoader
from torch.utils.data import random_split

valid_size = 128
batch_size = 32
valid, train = random_split(sampler_dataset, [valid_size, 5000 - valid_size])
train_generator = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)


def test_evaluate(discriminator, device, step, save = True):
    loss = nn.BCEWithLogitsLoss()
    loss_value = 0
    valid_sampler = iter(valid_generator)
    for i in range(valid_size):
        with torch.no_grad():
            imgs, labels = valid_sampler.next()
            imgs = to_tensor(imgs, device)
            labels = to_tensor(labels, device)
            logits = discriminator(imgs)
            loss_value += loss(logits, labels)
    writer.add_scalar("Loss/valid", loss_value, step)
    print(f"validation loss: {loss_value.item()} (for scale: {loss_value.item() / (valid_size)}")



if __name__ == "__main__":
    #TODO: missing augmentations, weird structure, etc
    #TODO: these are the parameters you can modify
    experiment = "discriminator"
    load_model = False

    num_training_steps = 10000
    path = f"G:\\Desktop\\Working Repository\\CRUMPL\\experiments\\{experiment}"

    writer = SummaryWriter(path)  # you can specify logging directory



    discriminator = Discriminator((3, 128, 128))
    print("done generating and loading models")
    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        discriminator = discriminator.cuda()
        # decoder = decoder.cuda()
    else:
        device = "cpu"

    # if load_model:
    #     checkpoint = 100000
    #     encoder.load_state_dict(torch.load(f'{path}/model_weights_encoder_{checkpoint}.pth'))
    #     # decoder.load_state_dict(torch.load(f'{path}/model_weights_decoder_{checkpoint}.pth'))
    #     # test_evaluate(encoder, decoder, device, step = "TEST", save = True)
    #     quit()

    torch.autograd.set_detect_anomaly(True)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    loss = nn.BCEWithLogitsLoss()

    soft_make_dir(path)
    os.chdir(path)
    f = open("metrics_train.csv", "w", newline="")
    csv_train_writer = csv.writer(f)
    csv_train_writer.writerow(["step", "loss"])
    f = open("metrics_valid.csv", "w", newline="")
    csv_valid_writer = csv.writer(f)
    csv_valid_writer.writerow(["step", "MI", "MSE"])

    norm_mult = 1e-7
    train_sampler = iter(train_generator)
    for i in range(num_training_steps + 1):
        if i % 151 == 0:
            train_sampler = iter(train_generator)

        if i % 200 == 0:
            writer.flush()
            print("eval time!")
            torch.save(discriminator.state_dict(), f"model_weights_encoder_{i}.pth")  # saves everything from the state dictionary
            # torch.save(decoder.state_dict(), f"model_weights_decoder_{i}.pth")  # saves everything from the state dictionary
            test_evaluate(discriminator, device, step = i, save = True)
        beg = time.time()
        imgs, labels = train_sampler.next()
        imgs = to_tensor(imgs, device)
        labels = to_tensor(labels, device)
        logits = discriminator(imgs)
        # import pdb
        # pdb.set_trace()
        class_loss = loss(logits, labels)

        if i % 25 == 0:
            print(i, " ", to_numpy(class_loss))

        discriminator_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        class_loss.backward()
        discriminator_optimizer.step()
        # decoder_optimizer.step()
        csv_train_writer.writerow([i, class_loss.detach().cpu().item()])
        writer.add_scalar("Loss/train_loss", class_loss, i)
        # print(time.time() - beg)
    writer.close()
