import librosa
import os
import numpy as np
import soundfile as sf
import threading
import scipy
import pickle
import csv
import torch
import matplotlib.pyplot as plt
# from train_unsupervised import load_and_freeze_classifier, make_classifier

def read_audio_spectum(filename):
    N_FFT = 1024
    x, fs = librosa.load(filename)  # Duration=58.05 so as to make sizes convenient
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, fs

#expects files in format generated_{instrument}_{number}.wav and truth

num_trials = 25

def generate_stats(base_dir, label):
    instruments_dict = {}
    instruments = ["distortion", "harp", "harpsichord", "piano", "timpani"]

    # classifier_load_dir = "experiments_classifier/audio_classifier_smaller/evaluation_15000/classifier_params_15000.pt"
    # classifier = make_classifier()
    # classifier = load_and_freeze_classifier(classifier, classifier_load_dir)
    # classifier = classifier.cuda()
    # confusion_matrix = np.zeros((len(instruments), len(instruments)))

    for j, instrument in enumerate(instruments):
        print(instrument)
        for i in range(num_trials):
            print(i)
            truth_file = f"{base_dir}/truth_{instrument}_{i}.wav"
            gen_file = f"{base_dir}/generated_{instrument}_{i}.wav"
            s_truth, _ = read_audio_spectum(truth_file)
            s_gen, _ = read_audio_spectum(gen_file)

            # s_gen_cuda = np.expand_dims(s_gen, axis=0)
            # s_gen_cuda = np.expand_dims(s_gen_cuda, axis=0)
            # s_gen_cuda = torch.as_tensor(s_gen_cuda, device="cuda")

            # prediction = classifier.forward(s_gen_cuda, None)
            # confusion_matrix[j][torch.argmax(prediction)] += 1

            if instrument not in instruments_dict:
                instruments_dict[instrument] = list()
            instruments_dict[instrument].append(np.mean((s_gen - s_truth)**2))

    means_dict = {}
    errors_dict = {}

    with open(f"stats{label}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["instrument", "mean", "standard error"])
        for instrument in instruments_dict:
            means_dict[instrument] = np.mean(np.asarray(instruments_dict[instrument]))
            errors_dict[instrument] = np.std(np.asarray(instruments_dict[instrument])) / 5  # sqrt of 25
            writer.writerow([instrument, means_dict[instrument], errors_dict[instrument]])

    return means_dict, errors_dict


if __name__ == "__main__":
    ind = np.arange(5) #number of instruments
    means_dict_list = list()
    errors_dict_list = list()
    labels_list = ["MidiStyle", "Style One-Hot", "No Residual", "No Batch Norm", "No Style Signal"]
    # labels_list = ["Inferred", "Ground Truth"]
    base_dir_list = ["experiments/piano_fullmodel_unsupervised_batchnorm/evaluation/evaluation_15000",
                     "experiments/piano_fullmodel_unsupervised_truthstyle/evaluation/evaluation_15000",
                    "experiments/piano_fullmodel_unsupervised_ablation_nopassthrough/evaluation/evaluation_15000",
                     "experiments/piano_fullmodel_unsupervised_ablation_nobatchnorm/evaluation/evaluation_15000",
                     "experiments/piano_fullmodel_unsupervised_ablation_nostyle/evaluation/evaluation_15000",

                     # "Neural-Style-Transfer-Audio-master/samples"
                     ]



    width = 0.75 / len(labels_list)
    bar_list = list()
    i = 0
    for label, base_dir in zip(labels_list, base_dir_list):
        means, errors = generate_stats(base_dir, label)
        bar_list.append(plt.bar(ind - (len(labels_list) / 2) * width + i * width, means.values(), width=width))
        plt.errorbar(ind - (len(labels_list) / 2) * width + i * width, means.values(), yerr=errors.values(), fmt='none', ecolor='red',
                     capsize=2)
        i += 1
    plt.title("Model Ablations (25 trials)")
    plt.xticks(ticks=ind, labels=means.keys())
    plt.ylabel("MSE Spectrogram Difference")
    plt.legend(bar_list, labels_list)
    plt.show()
    plt.savefig(f"Ablations2.pdf")
    plt.savefig(f"Ablations2.png")

