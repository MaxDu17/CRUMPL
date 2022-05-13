import pickle
import numpy as np
import torch
from torch import nn
import os
from shutil import copyfile
import matplotlib.pyplot as plt
from old.ConvAE import ConvAE, create_network
from torch.utils.tensorboard import SummaryWriter
sampler_dataset = pickle.load(open("../dataset_10000.pkl", "rb"))
print("done loading data")

# TODO: add csv logging on top of tensorboard because it's not working

from torch.utils.data import DataLoader
from torch.utils.data import random_split

valid_size = 128
valid, train = random_split(sampler_dataset, [valid_size, 5000 - valid_size])
train_generator = DataLoader(train, batch_size=4, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
plt.ion() #needed to prevent show() from blocking
# plt.figure(figsize=(15,25))




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
    # should save the model
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
            proposed_smooth, _ = generator_model(crumpled) #sanity check
            loss_value += loss(smooth, proposed_smooth)

            hist, value = generate_mutual_information(smooth.cpu().detach().numpy(), proposed_smooth.cpu().detach().numpy(), hist = True)
            hist_log = np.zeros(hist.shape)
            non_zeros = hist != 0
            hist_log[non_zeros] = np.log(hist[non_zeros])

            MI_value += value
            MI_base += generate_mutual_information(smooth.cpu().detach().numpy(), smooth.cpu().detach().numpy())
            MI_low += generate_mutual_information(smooth.cpu().detach().numpy(), crumpled.cpu().detach().numpy())
            if i == random_selection:
                visualize(to_numpy(crumpled[0]), to_numpy(smooth[0]), to_numpy(proposed_smooth[0]), hist_log, save = save, step = step, visible = True)
    writer.add_scalar("Loss/valid", loss_value, step)
    writer.add_scalar("Loss/valid_MI", MI_value, step)
    print(f"Mutual information value (higher better): {MI_value}, which is upper bounded by {MI_base} and lower bounded by {MI_low}")
    print(f"validation loss: {loss_value.item()} (for scale: {loss_value.item() / (valid_size)}")

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
    experiment = "baseline_autoencoder_with_logging"
    load_model = False

    num_training_steps = 50000
    path = f"G:\\Desktop\\Working Repository\\CRUMPL\\experiments\\{experiment}"

    writer = SummaryWriter(path)  # you can specify logging directory

    generator_model = make_generator()
    print("done generating and loading models")
    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        generator_model = generator_model.cuda()
    else:
        device = "cpu"

    if load_model:
        checkpoint = 100000
        generator_model.load_state_dict(torch.load(f'{path}/model_weights_{checkpoint}.pth'))
        test_evaluate(generator_model, device, step = "TEST", save = True)
        quit()

    AE_optimizer = torch.optim.Adam(generator_model.parameters(), lr=1e-3)
    loss = nn.MSELoss()

    soft_make_dir(path)
    copyfile("train_vanilla_autoencoder.py", f"{path}/train_vanilla_autoencoder.py")
    copyfile("ConvAE.py", f"{path}/ConvAE.py")
    os.chdir(path)

    norm_mult = 1e-7
    train_sampler = iter(train_generator)
    for i in range(num_training_steps + 1):
        if i % 1217 == 0:
            train_sampler = iter(train_generator)
        if i % 1000 == 0:
            writer.flush()
            print("eval time!")
            torch.save(generator_model.state_dict(), f"model_weights_{i}.pth")  # saves everything from the state dictionary
            test_evaluate(generator_model, device, step = i, save = True)
        # beg = time.time()
        crumpled, smooth = train_sampler.next()
        # print("\t" + str(time.time() - beg))
        # beg = time.time()
        crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
        smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
        predicted_smooth, _ = generator_model.forward(crumpled) # sanity check: can we recreate?
        encoding_loss = loss(smooth, predicted_smooth) #+ norm_mult * torch.sum(torch.abs(out))
        if i % 25 == 0:
            print(i, " ", encoding_loss.cpu().detach().numpy())
        AE_optimizer.zero_grad()
        encoding_loss.backward()
        AE_optimizer.step()
        writer.add_scalar("Loss/train_loss", encoding_loss, i)
        # print(time.time() - beg)
    writer.close()
