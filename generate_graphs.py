import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

os.chdir("./experiments")
parameter = "Inception" #MI,MSE,Inception

values = list()
trials = ["autoencoder_baseline", "U_Net_baseline",  "Pix2Pix_4", "Pix2Pix_8", "Pix2Pix_16", "cyclegan_basic"]
names = ["AE Baseline", "U-Net", "Pix2Pix 4", "Pix2Pix 8", "Pix2Pix 16", "CycleGAN"]

for folder in trials:
    try:
        df = pd.read_csv(f'{folder}/metrics_test.csv', index_col=0)
        values.append(df[parameter].to_numpy()[0])
    except:
        print("skipping ", folder)


x_data = np.arange(1, len(trials) + 1) # np.asarray([1, 2, 3])
x_labels = names
y_data = values

fig, ax = plt.subplots(figsize=(9,5))
#this is a demonstration of two bars. Feel free to remove one of the bars
bar_object = ax.bar(x_data, y_data, width = 0.6)
#this is a demonstration of the error bar feature. Feel free to remove
# error_bars = ax.errorbar(x_data - 0.15, y_data, yerr = [1, 1, 1], fmt = 'none', ecolor="red", capsize = 2)
# ax.set_yticks(ticks = [1, 2, 3, 4, 5, 7]) #showing you what you can do with the ticks
ax.set_ylabel(f"{parameter}")
plt.xticks(ticks = x_data, labels = x_labels) #you must ust the plt here
ax.set_xlabel("Models")
ax.set_title(f"{parameter} Score Across Models")
for bar in bar_object:
    bar.set_color("orange") #sets the first bar orange

# fig.savefig("test.png")
# fig.savefig("test.pdf") #save as pdf for paper-ready presentation
plt.show()



