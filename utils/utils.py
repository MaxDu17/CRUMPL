import numpy as np
import torch
import os
from matplotlib import pyplot as plt

def generate_plot(ncols):
    fig, ax_tuple = plt.subplots(ncols=4)
    plt.ion()  # needed to prevent show() from blocking
    return (fig, ax_tuple)

def visualize(axis_objects, img_list, img_names, save = False, step = 'no-step', visible = True):
    fig, ax_tuple = axis_objects
    for ax in ax_tuple:
        ax.clear()
        ax.set_axis_off()

    for name, img, ax in zip(img_names, img_list, ax_tuple):
        ax.title.set_text(name)
        if len(img.shape) == 3:
            ax.imshow(np.transpose(img, (1, 2, 0)))
        else:
            ax.imshow(img)

    if visible:
        plt.show()
    plt.pause(1)
    if save:
        plt.savefig(f"{step}.png", bbox_inches='tight',pad_inches = 0)



def soft_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("directory already exists!")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_tensor(arr, device):
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


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
