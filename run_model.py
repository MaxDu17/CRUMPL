import librosa
import pickle
import numpy as np
import random
import torch
from torch import nn
import os
import soundfile as sf
from shutil import copyfile
import threading
import matplotlib.pyplot as plt
from ConvAE import ConvAE, create_network, accuracy_1_min_mab, normalized_loss
from pipeline_whole import SampleLibrary

def read_audio_spectum(filename):
    N_FFT = 1024
    x, fs = librosa.load(filename)  # Duration=58.05 so as to make sizes convenient
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    # input(S.shape)
    # nfft / 2 + 1, by
    # S = np.pad(S, ((0, 0),(0, 513 - 431)), mode = "constant", constant_values = 0)
    return S, fs

def phase_reconstructor(output, filename):
    N_FFT = 1024
    output = np.squeeze(output)
    a = np.zeros_like(output)
    a = np.exp(output) - 1
    print("phase reconstruction")
    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        if i % 10 == 0:
            print("\t", i)
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
    sf.write(filename, x, 22050)

def make_generator():
    feature_maps = 64
    depth = 8
    pooling_freq = 1e100  # large number to disable pooling layers
    strided_conv_freq = 2
    strided_conv_feature_maps = 64
    input_dim = (1, 513, 513)
    style_hidden = 32
    num_style_interferences = 3

    CONV_ENC_BLOCK = [("conv1", feature_maps), ("relu1", None)]
    CONV_ENC_LAYERS = create_network(CONV_ENC_BLOCK, depth,
                                        pooling_freq=pooling_freq,
                                        strided_conv_freq=strided_conv_freq,
                                        strided_conv_channels=strided_conv_feature_maps,
#                                         batch_norm_freq = 1e100
                                        )
    CONV_ENC_NW = CONV_ENC_LAYERS
    model = ConvAE(input_dim, enc_config=CONV_ENC_NW, num_instruments = 5, style_hidden_size = style_hidden, num_style_interferences =
                   num_style_interferences)
    return model

def make_classifier():
    feature_maps = 8
    depth = 12
    pooling_freq = 1e100  # large number to disable pooling layers
    strided_conv_freq = 2
    strided_conv_feature_maps = 8
    code_size = 5
    input_dim = (1, 513, 513)
    CONV_ENC_BLOCK = [("conv1", feature_maps), ("relu1", None)]
    CONV_ENC_LAYERS = create_network(CONV_ENC_BLOCK, depth,
                                     pooling_freq=pooling_freq,
                                     strided_conv_freq=strided_conv_freq,
                                     strided_conv_channels=strided_conv_feature_maps,
                                     )

    CONV_ENC_NW = CONV_ENC_LAYERS + [("flatten1", None), ("linear1", code_size), ("softmax1", None)]

    model = ConvAE(input_dim, enc_config=CONV_ENC_NW, disable_decoder = True)
    return model


reconstructor_thread_list = list()

def generate_output(generator_model, classifier_frozen, data, style):
    # expect style to be 1 x 513 x 513
    # expect data to be 513 x something

    data_dims = data.shape[0]
    data_length = data.shape[1]
    output = np.zeros_like(data)
    # input(output[:, 3 : 516].shape)

    generator_model.eval()
    soft_make_dir(f"evaluation_whole_{step}")
    os.chdir(f"evaluation_whole_{step}")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    for i in range(0, data_length - data_dims, data_dims):
        print(f"{i} out of {data_length}")
        selected_data = data[:, i : i + data_dims]
        # style = np.expand_dims(style, axis = 0)
        # style = torch.as_tensor(style, device=device)

        selected_data = torch.as_tensor(selected_data, device=device)
        selected_data = torch.unsqueeze(selected_data, dim = 0)
        selected_data = torch.unsqueeze(selected_data, dim = 0)
        # style_vec = classifier_frozen.forward(style, None)

        # ["distortion", "harp", "harpsichord", "piano", "timpani"]

        style_vec = torch.as_tensor([0, 0, 0, 0 ,1], dtype = torch.float32, device = device)
        style_vec = torch.unsqueeze(style_vec, dim = 0)

        out, code = generator_model.forward(selected_data, style_vec)
        if torch.cuda.is_available():
            out = out.cpu().detach()

        output[:, i : i + data_dims] = out[0][0]

    phase_reconstructor(output,  f"generated_pathetique_timpani.wav")

def soft_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("directory already exists!")


def load_and_evaluate(generator_model, classifier_frozen, file_name, step, data, style):
    generator_model.load_state_dict(torch.load(file_name))
    generator_model.load_feature_maps(step)
    print("done loading!")
    soft_make_dir("evaluation")
    os.chdir("evaluation")
    generate_output(generator_model, classifier_frozen, data, style) #turn to true
    print("done!!")

def load_and_freeze_classifier(classifier, load_dir):
    classifier.load_state_dict(torch.load(load_dir))
    for p in classifier.parameters():
        p.requires_grad = False
    print("done loading and freezing classifier")
    return classifier

if __name__ == "__main__":
    classifier_load_dir = "experiments_classifier/audio_classifier_smaller/evaluation_15000/classifier_params_15000.pt"
    generator_model = make_generator()
    classifier = make_classifier()
    classifier_frozen = load_and_freeze_classifier(classifier, classifier_load_dir)
    print("done generating and loading models")

    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        classifier_frozen = classifier_frozen.cuda()
        generator_model = generator_model.cuda()
    else:
        device = "cpu"

    data_file = "music/pathetique_2_piano.wav"
    style_file = "music/pathetique_2_harpsichord.wav"

    s_data, sr = read_audio_spectum(data_file)
    s_style, sr = read_audio_spectum(style_file)

    s_style = s_style[:, 200 : 200 + 513]


    exp_dir =  "piano_fullmodel_unsupervised_batchnorm"
    step = 15000
    file_name = f"ae_params_{step}.pt"
    tests_per_instrument = 25 #chanve back
    try:
        os.chdir(f"experiments/{exp_dir}")
    except:
        print("this experiment doesn't exist!")
        quit()
    load_and_evaluate(generator_model, classifier_frozen, file_name, step, s_data, s_style)
    quit()