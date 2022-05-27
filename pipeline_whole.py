import numpy as np
import os
from torch.utils.data import IterableDataset
import imageio
from tqdm import tqdm
import cv2

class CrumpleLibrary(IterableDataset):
    def __init__(self, base_directory, number_images = 1000):
        self.base_directory = base_directory
        file_list = sorted(os.listdir(self.base_directory))

        self.crumpled_list = list()
        self.smooth_list = list()
        self.num_items = number_images

        for i in tqdm(range(0, self.num_items, 2)):
            crumpled = imageio.imread(self.base_directory + file_list[i])
            crumpled = cv2.resize(crumpled, (128, 128))
            smooth = imageio.imread(self.base_directory + file_list[i + 1])
            smooth = cv2.resize(smooth, (128, 128))

            # remove alpha channel if it exists
            if crumpled.shape[2] == 4:
                crumpled = crumpled[:, :, :3]
            if smooth.shape[2] == 4:
                smooth = smooth[:, :, :3]

            self.crumpled_list.append(crumpled)
            self.smooth_list.append(smooth)

        self.mode = "pos_neg_sample"

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return self.num_items // 2

    def single_sample(self, idx):
        return np.transpose(np.array(self.crumpled_list[idx] / 255.), axes = (2, 0, 1)), \
               np.transpose(np.array(self.smooth_list[idx] / 255.), axes = (2, 0, 1))

    def pos_neg_sample(self, idx):
        POSITIVE_PROB = 0.5
        if np.random.rand() > POSITIVE_PROB:
            other_index = np.random.randint(0, self.__len__())
            # picking which list you end up picking from
            if np.random.rand() > 0.5:
                selected_list = self.smooth_list
            else:
                selected_list = self.crumpled_list
            return np.transpose(np.array(self.crumpled_list[idx] / 255.), axes=(2, 0, 1)), \
                   np.transpose(np.array(selected_list[other_index] / 255.), axes=(2, 0, 1)), -1
        else:
            return np.transpose(np.array(self.crumpled_list[idx] / 255.), axes=(2, 0, 1)), \
                   np.transpose(np.array(self.smooth_list[idx] / 255.), axes=(2, 0, 1)), 1

    def classifier_sample(self, idx):
        if np.random.rand() > 0.5:
            selected_list = self.smooth_list
            label = np.array([1])
        else:
            selected_list = self.crumpled_list
            label = np.array([0])
        return np.transpose(np.array(selected_list[idx] / 255.), axes = (2, 0, 1)), label

    def __getitem__(self, idx):
        if self.mode == "pos_neg_sample":
            return self.pos_neg_sample(idx)
        elif self.mode == "single_sample":
            return self.single_sample(idx)
        elif self.mode == "classifier_sample":
            return self.classifier_sample(idx)
        else:
            raise Exception("invalid type!")
        # later, we will delegate to a process function

