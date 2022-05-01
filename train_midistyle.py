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
sampler_dataset = pickle.load(open("simple_dataset.pkl", "rb"))
print("done loading data")

def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

sampler_dataset_iterable = iter(torch.utils.data.DataLoader(sampler_dataset,
                                                      batch_size=8,
                                                      num_workers=0,
                                                      # pin_memory=True,
                                                      worker_init_fn=worker_init_fn)) # what is pin memory?

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
plt.ion() #needed to prevent show() from blocking
# plt.figure(figsize=(15,25))

def visualize(content, style, output, save = False, name = None, step = 'no-step', visible = True):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax1.title.set_text(f"Ground Truth")
    ax1.imshow(content)
    ax2.title.set_text(f"Style ({name})")
    ax2.imshow(style)
    ax3.title.set_text(f"Output")
    ax3.imshow(output)
    if visible:
        plt.show()
    plt.pause(1)
    if save and name is None:
        plt.savefig("test.png")
    elif save:
        plt.savefig(f"{name}_{step}.png", bbox_inches='tight',pad_inches = 0)

def phase_reconstructor(output, filename):
    N_FFT = 1024
    output = np.squeeze(output)
    a = np.zeros_like(output)
    a = np.exp(output) - 1
    print("phase reconstruction")
    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        if i % 100 == 0:
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

def test_evaluate(generator_model, classifier_frozen, num_tests_per_instrument, step, save = True, diff_target = False):
    instruments = ["distortion", "harp", "harpsichord", "piano", "timpani"]
    averages_dict = {}
    style = None
#     input(generator_model.state_dict().keys())
    generator_model.eval()
    if save:
        print("saving model!")
        generator_model.save_feature_maps(step)
        torch.save(generator_model.state_dict(), f"ae_params_{step}.pt")

    soft_make_dir(f"evaluation_{step}")
    os.chdir(f"evaluation_{step}")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    for instrument in instruments:
        print(instrument)
        for i in range(num_tests_per_instrument):
            print(f"\t{i}")
            if diff_target:
                data, target, style = sampler_dataset.sampleTest(test_instrument=instrument)
                style = np.expand_dims(style, axis = 0)
                style = torch.as_tensor(style, device=device)

            else:
                data, target, one_hot = sampler_dataset.samplePair(test = True, test_instrument = instrument)

            data = np.expand_dims(data, axis = 0)
            target = np.expand_dims(target, axis = 0)
            data = torch.as_tensor(data, device=device)
            target = torch.as_tensor(target, device=device)

            style = style if diff_target else target

            style_vec = classifier_frozen.forward(style, None)

            out, code = generator_model.forward(data, style_vec)

            if torch.cuda.is_available():
                out = out.cpu().detach()
                target = target.cpu().detach()
                data = data.cpu().detach()
                style = style.cpu().detach()

            encoding_loss = ((out - target)**2).mean()
            if instrument not in averages_dict.keys():
                averages_dict[instrument] = 0
            averages_dict[instrument] += encoding_loss.numpy()
            print("saving plot!")
            visualize(data[0][0], style[0][0], out[0][0], save=True, name=instrument, step = i, visible = False)

            print("saving generating audios")
            t_gen = threading.Thread(target=phase_reconstructor, args=(out.numpy()[0], f"generated_{instrument}_{i}.wav"))
            t_truth = threading.Thread(target=phase_reconstructor, args=(target.numpy()[0], f"truth_{instrument}_{i}.wav"))
            t_gen.start()
            t_truth.start()
            reconstructor_thread_list.append(t_gen)
            reconstructor_thread_list.append(t_truth)
        averages_dict[instrument] /= num_tests_per_instrument

        for t in reconstructor_thread_list: #multithreading per trial
            t.join()
    print(averages_dict)
    import json
    json = json.dumps(averages_dict)
    f = open("losses.json", "w")
    f.write(json)
    f.close()
    os.chdir("../")
    generator_model.train()

def soft_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print("directory already exists!")


def load_and_evaluate(generator_model, classifier_frozen, file_name, num_tests_per_instrument, step):
    generator_model.load_state_dict(torch.load(file_name))
    generator_model.load_feature_maps(step)
#     for key, value in generator_model.state_dict().items():
#         print(f"key {key}, value {value.shape}")
#     input("asdf")
    print("done loading!")
    soft_make_dir("evaluation")
    os.chdir("evaluation")
    test_evaluate(generator_model, classifier_frozen, num_tests_per_instrument = num_tests_per_instrument, step = step, save = False,
                  diff_target = True) #turn to true
    print("done!!")

def load_and_freeze_classifier(classifier, load_dir):
    classifier.load_state_dict(torch.load(load_dir))
    for p in classifier.parameters():
        p.requires_grad = False
    print("done loading and freezing classifier")
    return classifier

if __name__ == "__main__":
    #TODO: these are the parameters you can modify
    classifier_load_dir = "experiments_classifier/audio_classifier_smaller/evaluation_15000/classifier_params_15000.pt"
    experiment = "piano_fullmodel_unsupervised_truthstyle"
    num_training_steps = 15000
    path = f"experiments/{experiment}"
    load_test = True

    generator_model = make_generator()
    classifier = make_classifier()
    # input(classifier)
    classifier_frozen = load_and_freeze_classifier(classifier, classifier_load_dir)
    print("done generating and loading models")

    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        classifier_frozen = classifier_frozen.cuda()
        generator_model = generator_model.cuda()
    else:
        device = "cpu"

    AE_optimizer = torch.optim.Adam(generator_model.parameters(), lr=1e-3)
    loss = nn.MSELoss()

    if load_test:
        exp_dir =  "piano_fullmodel_unsupervised_truthstyle"
        step = 15000
        file_name = f"ae_params_{step}.pt"
        tests_per_instrument = 25 #chanve back
        try:
            os.chdir(f"experiments/{exp_dir}")
        except:
            print("this experiment doesn't exist!")
            quit()
        load_and_evaluate(generator_model, classifier_frozen, file_name, num_tests_per_instrument = tests_per_instrument, step = step)
        quit()

    soft_make_dir(path)
    copyfile("train_unsupervised.py", f"{path}/train_unsupervised.py")
    copyfile("ConvAE.py", f"{path}/ConvAE.py")
    copyfile("pipeline_whole.py", f"{path}/pipeline_whole.py")
    os.chdir(path)

    norm_mult = 1e-7
    for i in range(num_training_steps + 1):
        x, target, one_hot = sampler_dataset_iterable.next()
        one_hot = torch.as_tensor(one_hot, device=device, dtype = torch.float32)
        x = torch.as_tensor(x, device=device)
        target = torch.as_tensor(target, device=device)
        style_vec = classifier_frozen.forward(target, None)
#         out, code = generator_model.forward(x, one_hot) #CHANGE BACK
        out, code = generator_model.forward(x, style_vec)
        if i % 1500 == 0:
            print("eval time!")
            test_evaluate(generator_model, classifier_frozen, num_tests_per_instrument = 5, step = i, save = True, diff_target = True)
        # print("l1 loss: ", norm_mult * torch.sum(torch.abs(out)).cpu().detach().numpy())
        # print("norm loss: " , norm_mult * torch.norm(out).cpu().detach().numpy())
        # encoding_loss = normalized_loss(out, target)
        encoding_loss = loss(out, target) #+ norm_mult * torch.sum(torch.abs(out))
        print(i, " ", encoding_loss.cpu().detach().numpy())
        AE_optimizer.zero_grad()
        encoding_loss.backward()
        AE_optimizer.step()
