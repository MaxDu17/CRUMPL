import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

os.chdir("./experiments")
parameter = "MSE" #MI,MSE,Inception

values = list()
stdev = list()
trials = ["autoencoder_baseline", "U_Net_baseline", "Pix2Pix_4", "Pix2Pix_8", "Pix2Pix_16",  "Pix2Pix_32"]#, "cyclegan_basic"]
names = ["AE Baseline", "U-Net", "Pix2Pix 4", "Pix2Pix 8", "Pix2Pix 16", "Pix2Pix 32"]#, "CycleGAN"]

for folder in trials:
    try:
        df = pd.read_csv(f'{folder}/metrics_test.csv', index_col=0)
        values.append(np.mean(df[parameter].to_numpy()))
        stdev.append(np.std(df[parameter].to_numpy()) / np.sqrt(1000))
    except:
        print("skipping ", folder)


x_data = np.arange(1, len(trials) + 1) # np.asarray([1, 2, 3])
x_labels = names
y_data = values
print(values)

fig, ax = plt.subplots(figsize=(9,5))
bar_object = ax.bar(x_data, y_data, width = 0.6)
#this is a demonstration of the error bar feature. Feel free to remove
error_bars = ax.errorbar(x_data, y_data, yerr = stdev, fmt = 'none', ecolor="black", capsize = 2)
ax.set_ylabel(f"{parameter}")
# ax.set_ylim((2000, 6000)) # for Inception
plt.xticks(ticks = x_data, labels = x_labels) #you must ust the plt here
ax.set_xlabel("Models")
ax.set_title(f"{parameter} Score Across Models")
for bar in bar_object:
    bar.set_color("gray")

# fig.savefig("test.png")
fig.savefig("MSE.pdf") #save as pdf for paper-ready presentation
plt.show()



