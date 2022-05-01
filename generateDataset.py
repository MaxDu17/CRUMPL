from pipeline_whole import CrumpleLibrary
import pickle

generatedLibrary = CrumpleLibrary(base_directory = "data/paired_data/", number_images = 10000)

with open("dataset_10000.pkl", "wb") as f:
    pickle.dump(generatedLibrary, f, protocol=4)
#
# with open("simple_dataset.pkl", "rb") as f:
#     generatedLibrary = pickle.load(f)

# single = generatedLibrary.samplePair(test = True)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.imshow(single[0])
# plt.show()
