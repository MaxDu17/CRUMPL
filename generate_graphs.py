import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

os.chdir("./experiments")
parameter = "Inception" #MI,MSE,Inception

values = list()
stdev = list()
trials = ["autoencoder_baseline", "U_Net_baseline", "Pix2Pix_4", "Pix2Pix_8", "Pix2Pix_16",  "Pix2Pix_32",  "Pix2Pix_64",  "Pix2Pix_128"] #, "cyclegan_basic"]
names = ["AE Baseline", "U-Net", "Pix2Pix 94", "Pix2Pix 46", "Pix2Pix 22", "Pix2Pix 10", "Pix2Pix 4", "Pix2Pix 1"] #, "CycleGAN"]

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

for i in range(len(values)):
    v = round(values[i] * 100, 2)
    s = round(stdev[i] * 100, 2)
    print(f"{v} $\\pm$ {s}")
# print(values)
# print(stdev)

fig, ax = plt.subplots(figsize=(11,5))
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
fig.savefig("inception.pdf") #save as pdf for paper-ready presentation
plt.show()



