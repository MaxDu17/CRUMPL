from pipeline_whole import CrumpleLibrary
import pickle

# uncomment to generate
# generatedLibrary = CrumpleLibrary(base_directory = "data/paired_data/", number_images = 10000)
#
# with open("dataset_10000.pkl", "wb") as f:
#     pickle.dump(generatedLibrary, f, protocol=4)

# uncomment to test
# with open("dataset_10000.pkl", "rb") as f:
#     generatedLibrary = pickle.load(f)
#
# crumpled, smooth = generatedLibrary[2]
# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(ncols = 2)
# ax1.imshow(crumpled)
# ax2.imshow(smooth)
# plt.show()
# input("hold") #to check memory
