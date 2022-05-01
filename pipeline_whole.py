import numpy as np
import os
from torch.utils.data import IterableDataset
import imageio
from tqdm import tqdm

class CrumpleLibrary(IterableDataset):
    def __init__(self, base_directory, number_images = 1000):
        self.base_directory = base_directory
        file_list = sorted(os.listdir(self.base_directory))

        self.crumpled_list = list()
        self.smooth_list = list()
        self.num_items = number_images

        for i in tqdm(range(0, self.num_items, 2)):
            crumpled = imageio.imread(self.base_directory + file_list[i])
            smooth = imageio.imread(self.base_directory + file_list[i + 1])
            self.crumpled_list.append(crumpled)
            self.smooth_list.append(smooth)

    def __len__(self):
        return self.num_items // 2

    def __getitem__(self, idx):
        # later, we will delegate to a process function
        return np.transpose(np.array(self.crumpled_list[idx] / 255.), axes = (2, 0, 1)), \
               np.transpose(np.array(self.smooth_list[idx] / 255.), axes = (2, 0, 1))
