from pipeline_whole import CrumpleLibrary
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# uncomment to generate
generatedLibrary = CrumpleLibrary(base_directory = "data/paired_data_TEST/", number_images = 2000)

with open("frozen_datasets/dataset_49000_TEST.pkl", "wb") as f:
    pickle.dump(generatedLibrary, f, protocol=4)

# uncomment to test
# with open("frozen_datasets/dataset_49000_small.pkl", "rb") as f:
#     generatedLibrary = pickle.load(f)
#     generatedLibrary.set_mode('single_sample')
#
# fig, (ax1, ax2) = plt.subplots(ncols = 2)
# for i in range(len(generatedLibrary)):
#     for j in tqdm(range(len(generatedLibrary))):
#         _, smooth1 = generatedLibrary[i]
#         _, smooth2 = generatedLibrary[j]
#         if i != j and np.array_equal(smooth1, smooth2):
#             print(f"Found something! {i}, {j}")
#             ax1.imshow(np.transpose(smooth1, (1, 2, 0)))
#             ax2.imshow(np.transpose(smooth2, (1, 2, 0)))
#             plt.show()

# print(type)

# input("hold") #to check memory
