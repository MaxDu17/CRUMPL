import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from pipeline_whole import CrumpleLibrary

with open("frozen_datasets/dataset_49000_small.pkl", "rb") as f:
    generated_library = pickle.load(f)
    generated_library.set_mode('single_sample')

# generated_library = CrumpleLibrary(base_directory="data/val_blurred/image_exports/", number_images=102)
# generated_library.set_mode('single_sample')

hashes = np.zeros(len(generated_library))


def hash_arr(a):
    return np.sum(a)
    # hsh = 1
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         for k in range(a.shape[2]):
    #             hsh *= 10 * a[i, j, k]
    #             hsh += 7
    #             hsh %= (1e9 + 7)
    # return hsh


print("hashing...")
for i in tqdm(range(len(generated_library))):
    _, a = generated_library[i]
    hashes[i] = hash_arr(a)

idx_sort = np.argsort(hashes, kind='mergesort')
sorted_hashes = hashes[idx_sort]
vals, idx_start, count = np.unique(sorted_hashes, return_counts=True, return_index=True)
res = np.split(idx_sort, idx_start[1:])
vals = vals[count > 1]
res = list(filter(lambda x: x.size > 1, res))

# print(res)
fig, (ax1, ax2) = plt.subplots(ncols = 2)

for arr in res:
    i, j = arr[0], arr[1]
    assert i != j
    _, smooth1 =  generated_library[i]
    _, smooth2 = generated_library[j]
    if np.array_equal(smooth1, smooth2):
        print(f"ARRAYS EQUAL! uh oh. {i}, {j}")
        # display images here
        # ax1.imshow(np.transpose(smooth1, (1, 2, 0)))
        # ax2.imshow(np.transpose(smooth2, (1, 2, 0)))
        # plt.show()
