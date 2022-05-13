import pickle
import numpy as np
import random
import torch
from torch import nn
import os
import time
from shutil import copyfile
import matplotlib.pyplot as plt
from EncoderDecoder import Encoder, Decoder
import csv
from pipeline_whole import CrumpleLibrary
from torch.utils.tensorboard import SummaryWriter
sampler_dataset = pickle.load(open("dataset_10000.pkl", "rb"))
print("done loading data")

# TODO: add csv logging on top of tensorboard because it's not working

from torch.utils.data import DataLoader
from torch.utils.data import random_split

valid_size = 128
valid, train = random_split(sampler_dataset, [valid_size, 5000 - valid_size])
train_generator = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
plt.ion() #needed to prevent show() from blocking


def visualize(crumpled, smooth, output, MI_map, save = False, step = 'no-step', visible = True):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax1.title.set_text(f"Crumpled")
    ax1.imshow(np.transpose(crumpled, (1, 2, 0)))
    ax2.imshow(np.transpose(smooth, (1, 2, 0)))
    ax2.title.set_text(f"Smooth")
    ax3.title.set_text(f"Output")
    ax3.imshow(np.transpose(output, (1, 2, 0)))
    ax4.imshow(MI_map)
    if visible:
        plt.show()
    plt.pause(1)
    if save:
        plt.savefig(f"{step}.png", bbox_inches='tight',pad_inches = 0)

def make_generator():
    encoder = Encoder((3, 512, 512))
    decoder = Decoder((3, 512, 512))
    return encoder, decoder


def soft_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("directory already exists!")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def test_evaluate(encoder, decoder, device, step, save = True):
    loss = nn.MSELoss()
    random_selection = np.random.randint(valid_size)
    loss_value = 0
    MI_value = 0
    MI_base = 0
    MI_low = 0
    valid_sampler = iter(valid_generator)
    for i in range(valid_size):
        with torch.no_grad():
            crumpled, smooth = valid_sampler.next()
            crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
            smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
            encoding, activations = encoder(crumpled)
            proposed_smooth = decoder(encoding, activations)
            loss_value += loss(smooth, proposed_smooth)

            hist, value = generate_mutual_information(to_numpy(smooth), to_numpy(proposed_smooth), hist = True)
            hist_log = np.zeros(hist.shape)
            non_zeros = hist != 0
            hist_log[non_zeros] = np.log(hist[non_zeros])

            MI_value += value
            MI_base += generate_mutual_information(to_numpy(smooth), to_numpy(smooth))
            MI_low += generate_mutual_information(to_numpy(smooth), to_numpy(crumpled))
            if i == random_selection:
                visualize(to_numpy(crumpled[0]), to_numpy(smooth[0]), to_numpy(proposed_smooth[0]), hist_log, save = save, step = step, visible = True)
    writer.add_scalar("Loss/valid", loss_value, step)
    writer.add_scalar("Loss/valid_MI", MI_value, step)
    print(f"Mutual information value (higher better): {MI_value}, which is upper bounded by {MI_base} and lower bounded by {MI_low}")
    print(f"validation loss: {loss_value.item()} (for scale: {loss_value.item() / (valid_size)}")
    csv_valid_writer.writerow([step, MI_value, loss_value.item()])

def generate_mutual_information(img1, img2, hist = False):
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins = 20)
    joint_dist = hist_2d / np.sum(hist_2d)
    x_dist = np.sum(joint_dist, axis = 1)
    y_dist = np.sum(joint_dist, axis = 0)
    independent_dist = x_dist[:, None] * y_dist[None, :]
    non_zero_mask = joint_dist > 0
    if hist:
        return hist_2d, np.sum(joint_dist[non_zero_mask] * np.log(joint_dist[non_zero_mask] / independent_dist[non_zero_mask]))
    else:
        return np.sum(joint_dist[non_zero_mask] * np.log(joint_dist[non_zero_mask] / independent_dist[non_zero_mask]))

if __name__ == "__main__":
    #TODO: missing augmentations, weird structure, etc
    #TODO: these are the parameters you can modify
    experiment = "baseline_new"
    load_model = False

    num_training_steps = 10000
    path = f"G:\\Desktop\\Working Repository\\CRUMPL\\experiments\\{experiment}"

    writer = SummaryWriter(path)  # you can specify logging directory



    encoder, decoder = make_generator()
    print("done generating and loading models")
    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    else:
        device = "cpu"

    if load_model:
        checkpoint = 100000
        encoder.load_state_dict(torch.load(f'{path}/model_weights_encoder_{checkpoint}.pth'))
        decoder.load_state_dict(torch.load(f'{path}/model_weights_decoder_{checkpoint}.pth'))
        test_evaluate(encoder, decoder, device, step = "TEST", save = True)
        quit()

    torch.autograd.set_detect_anomaly(True)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss = nn.MSELoss()

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
            torch.save(encoder.state_dict(), f"model_weights_encoder_{i}.pth")  # saves everything from the state dictionary
            torch.save(decoder.state_dict(), f"model_weights_decoder_{i}.pth")  # saves everything from the state dictionary
            test_evaluate(encoder, decoder, device, step = i, save = True)
        beg = time.time()
        crumpled, smooth = train_sampler.next()
        crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
        smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
        embedding, activations = encoder.forward(crumpled) # sanity check: can we recreate?
        predicted_smooth = decoder(embedding, activations)
        encoding_loss = loss(smooth, predicted_smooth) #+ norm_mult * torch.sum(torch.abs(out))

        if i % 25 == 0:
            print(i, " ", encoding_loss.cpu().detach().numpy())

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoding_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        csv_train_writer.writerow([i, encoding_loss])
        writer.add_scalar("Loss/train_loss", encoding_loss, i)
        # print(time.time() - beg)
    writer.close()
