import pickle
import numpy as np
import random
import torch
from torch import nn
import os
import time
from shutil import copyfile
import matplotlib.pyplot as plt
from ConvAE import ConvAE, create_network, accuracy_1_min_mab, normalized_loss
from pipeline_whole import CrumpleLibrary
sampler_dataset = pickle.load(open("dataset_10000.pkl", "rb"))
print("done loading data")


from torch.utils.data import DataLoader
from torch.utils.data import random_split

valid_size = 128
valid, train = random_split(sampler_dataset, [valid_size, 5000 - valid_size])
train_generator = DataLoader(train, batch_size=4, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=4, shuffle=False, num_workers=0)


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
plt.ion() #needed to prevent show() from blocking
# plt.figure(figsize=(15,25))

def visualize(crumpled, smooth, output, save = False, step = 'no-step', visible = True):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax1.title.set_text(f"Crumpled")
    ax1.imshow(np.transpose(crumpled, (1, 2, 0)))
    ax2.imshow(np.transpose(smooth, (1, 2, 0)))
    ax2.title.set_text(f"Smooth")
    ax3.title.set_text(f"Output")
    ax3.imshow(np.transpose(output, (1, 2, 0)))
    if visible:
        plt.show()
    plt.pause(1)
    if save:
        plt.savefig(f"{step}.png", bbox_inches='tight',pad_inches = 0)

def make_generator():
    feature_maps = 64
    depth = 6
    pooling_freq = 1e100  # large number to disable pooling layers
    strided_conv_freq = 2
    strided_conv_feature_maps = 64
    input_dim = (3, 512, 512)

    CONV_ENC_BLOCK = [("conv1", feature_maps), ("relu1", None)]
    CONV_ENC_LAYERS = create_network(CONV_ENC_BLOCK, depth,
                                        pooling_freq=pooling_freq,
                                        strided_conv_freq=strided_conv_freq,
                                        strided_conv_channels=strided_conv_feature_maps,
                                        batch_norm_freq = 2
                                        )
    CONV_ENC_NW = CONV_ENC_LAYERS
    model = ConvAE(input_dim, enc_config=CONV_ENC_NW)
    return model


def soft_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("directory already exists!")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def test_evaluate(generator_model, device, step, save = True):
    loss = nn.MSELoss()
    random_selection = np.random.randint(valid_size // 4)
    loss_value = 0
    valid_sampler = iter(valid_generator)
    for i in range(valid_size // 4 - 1):
        with torch.no_grad():
            crumpled, smooth = valid_sampler.next()
            crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
            smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
            proposed_smooth, _ = generator_model(smooth) #sanity check
            loss_value += loss(smooth, proposed_smooth)
            if i == random_selection:
                visualize(to_numpy(crumpled[0]), to_numpy(smooth[0]), to_numpy(proposed_smooth[0]), save = save, step = step, visible = True)
    print(f"validation loss: {loss_value.item()}")


if __name__ == "__main__":
    #TODO: missing augmentations, weird structure, etc
    #TODO: these are the parameters you can modify
    experiment = "baseline_autoencoder"
    num_training_steps = 15000
    path = f"experiments/{experiment}"
    load_test = True

    generator_model = make_generator()
    print("done generating and loading models")

    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        generator_model = generator_model.cuda()
    else:
        device = "cpu"

    AE_optimizer = torch.optim.Adam(generator_model.parameters(), lr=1e-3)
    loss = nn.MSELoss()

    soft_make_dir(path)
    copyfile("train_vanilla_autoencoder.py", f"{path}/train_vanilla_autoencoder.py")
    copyfile("ConvAE.py", f"{path}/ConvAE.py")
    os.chdir(path)

    norm_mult = 1e-7
    train_sampler = iter(train_generator)
    for i in range(num_training_steps + 1):
        if i == 1217:
            train_sampler = iter(train_generator)
        if i % 100 == 0:
            print("eval time!")
            test_evaluate(generator_model, device, step = i, save = True)
        # beg = time.time()
        crumpled, smooth = train_sampler.next()
        # print("\t" + str(time.time() - beg))
        # beg = time.time()
        crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
        smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
        predicted_smooth, _ = generator_model.forward(smooth) # sanity check: can we recreate?
        encoding_loss = loss(smooth, predicted_smooth) #+ norm_mult * torch.sum(torch.abs(out))
        if i % 25 == 0:
            print(i, " ", encoding_loss.cpu().detach().numpy())
        AE_optimizer.zero_grad()
        encoding_loss.backward()
        AE_optimizer.step()
        # print(time.time() - beg)
