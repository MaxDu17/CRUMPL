from pipeline_whole import CrumpleLibrary
import pickle
import numpy as np

# uncomment to generate
# generatedLibrary = CrumpleLibrary(base_directory = "data/paired_data/", number_images = 98000)
#
# with open("dataset_49000_small.pkl", "wb") as f:
#     pickle.dump(generatedLibrary, f, protocol=4)

# uncomment to test
# with open("dataset_100000_small.pkl", "rb") as f:
#     generatedLibrary = pickle.load(f)
#
# crumpled, smooth, type = generatedLibrary[2]
# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(ncols = 2)
# ax1.imshow(np.transpose(crumpled, (1, 2, 0)))
# ax2.imshow(np.transpose(smooth, (1, 2, 0)))
# print(type)
# plt.show()
# input("hold") #to check memory
