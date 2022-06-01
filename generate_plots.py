import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

os.chdir("./experiments")
parameters = ["MSE", "Inception", "MI"] #MI,MSE,Inception

series_list = list()

fig, axs = plt.subplots(nrows=3, figsize=(4,7))
fig.subplots_adjust(hspace=0.5)

df = pd.read_csv(f'Pix2Pix_32/metrics_valid.csv', index_col=0)
line_list = list()
for parameter, ax, name in zip(parameters, axs, parameters):
    line = ax.plot(np.arange(101), df[parameter].to_numpy(), color = "orange")
    line_list.append(line)
    ax.set_title(name)
    # ax.set_yscale("log")

# ax.set_title(f"{parameter} Score Across Models")

# fig.savefig("test.png")
fig.savefig("LossPlots.pdf") #save as pdf for paper-ready presentation
plt.show()



