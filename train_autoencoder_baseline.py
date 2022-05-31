import pickle
from torch import nn
import time
from UNet.Models import Encoder, Decoder
import csv
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
import imagenet_inception_eval as inception_loss
from tqdm import tqdm

sampler_dataset = pickle.load(open("frozen_datasets/dataset_49000_small.pkl", "rb"))
sampler_dataset.set_mode("single_sample")

print("done loading data")

from torch.utils.data import DataLoader
from torch.utils.data import random_split

valid_size = 128
valid, train = random_split(sampler_dataset, [valid_size, 49000 - valid_size])
train_generator = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)

ax_objects = generate_plot(5) #creates plots that help with visualization

def make_generator():
    encoder = Encoder((3, 128, 128))
    decoder = Decoder((3, 128, 128))
    return encoder, decoder


def test_evaluate(encoder, decoder, device, sampler, step, writer = None, csv_writer = None, save = True, raw_output = None):
    loss = nn.MSELoss()
    random_selection = np.random.randint(valid_size)
    loss_value = 0
    MI_value = 0
    MI_base = 0
    MI_low = 0
    i_l = 0
    valid_sampler = iter(sampler)
    encoder.train(False)
    decoder.train(False)
    for i, (crumpled, smooth) in tqdm(enumerate(valid_sampler)):
        with torch.no_grad():
            crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
            smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
            encoding, _ = encoder(crumpled)
            proposed_smooth = decoder(encoding, None)
            loss_value += loss(smooth, proposed_smooth)

            if raw_output is not None:
                proposed_smooth_normalized = np.clip(to_numpy(proposed_smooth[0]), 0, 1)
                plt.imsave(raw_output + str(i) + ".png", np.transpose(proposed_smooth_normalized, (1, 2, 0)))

            hist, value = generate_mutual_information(to_numpy(smooth), to_numpy(proposed_smooth), hist = True)
            hist_log = np.zeros(hist.shape)
            non_zeros = hist != 0
            hist_log[non_zeros] = np.log(hist[non_zeros])

            MI_value += value
            MI_base += generate_mutual_information(to_numpy(smooth), to_numpy(smooth))
            MI_low += generate_mutual_information(to_numpy(smooth), to_numpy(crumpled))

            i_l += inception_loss.loss_on_batch(smooth, proposed_smooth)

            if i == random_selection:
                visualize(ax_objects,
                          [to_numpy(crumpled[0]), to_numpy(smooth[0]), to_numpy(proposed_smooth[0]), hist_log],
                          ["crumpled", "smooth", "output", "Mutual Info"], save=save, step=step, visible=True)

            if csv_writer is not None:
                csv_writer.writerow([step, value, loss(smooth, proposed_smooth).item(), inception_loss.loss_on_batch(smooth, proposed_smooth).item()])
    if writer is not None:
        writer.add_scalar("Loss/valid", loss_value, step)
        writer.add_scalar("Loss/valid_MI", MI_value, step)
        writer.add_scalar("Loss/valid_Inception", i_l, step)
    # if csv_writer is not None:
    #     csv_writer.writerow([step, MI_value, loss_value.item(), i_l.item()])

    print(f"Mutual information value (higher better): {MI_value}, which is upper bounded by {MI_base} and lower bounded by {MI_low}")
    print(f"validation loss: {loss_value.item()} (for scale: {loss_value.item() / (valid_size)}")

    encoder.train(True)
    decoder.train(True)

if __name__ == "__main__":
    experiment = "autoencoder_baseline"
    load_model = True

    num_training_steps = 50000
    path = os.getcwd() + f"/experiments/{experiment}"
    print(f"Experiment path: {path}")
    soft_make_dir(path)
    os.chdir(path)

    encoder, decoder = make_generator()
    print("done generating and loading models")
    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    else:
        device = "cpu"

    if load_model:
        checkpoint = 50000
        test_library = pickle.load(open("../../frozen_datasets/dataset_49000_TEST.pkl", "rb"))
        test_library.set_mode('single_sample')
        test, _ = random_split(test_library, [1000, 0])
        print("Done loading test data")
        test_generator = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
        encoder.load_state_dict(torch.load(f'{path}/model_weights_encoder_{checkpoint}.pth'))
        decoder.load_state_dict(torch.load(f'{path}/model_weights_decoder_{checkpoint}.pth'))
        def wrapper_uncrumpler(img):
            embedding, activations = encoder.forward(img)
            predicted_smooth = decoder(embedding, activations)
            return predicted_smooth

        f = open("metrics_test.csv", "w", newline="")
        csv_valid_writer = csv.writer(f)
        csv_valid_writer.writerow(["step", "MI", "MSE", "Inception"])
        test_evaluate(encoder, decoder, device, test_generator, step="TEST", csv_writer=csv_valid_writer, save=True,
                      raw_output=f"{path}/arbitrary_eval/")
        # encoder.train(False)
        # decoder.train(False)
        # run_through_model(wrapper_uncrumpler, "data/paired_data_TEST/", f"{path}/arbitrary_eval/", 128, device)
        quit()

    writer = SummaryWriter(path)  # you can specify logging directory
    torch.autograd.set_detect_anomaly(True)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss = nn.MSELoss()


    f = open("metrics_train.csv", "w", newline="")
    csv_train_writer = csv.writer(f)
    csv_train_writer.writerow(["step", "loss"])
    f = open("metrics_valid.csv", "w", newline="")
    csv_valid_writer = csv.writer(f)
    csv_valid_writer.writerow(["step", "MI", "MSE"])

    norm_mult = 1e-7
    train_sampler = iter(train_generator)
    for i in range(num_training_steps + 1):
        if i % 1527 == 0:
            train_sampler = iter(train_generator)

        if i % 200 == 0:
            writer.flush()
            print("eval time!")
            test_evaluate(encoder, decoder, device, valid_generator, step = i, writer = writer, csv_writer = csv_valid_writer, save = True)
        if i % 2500 == 0:
            torch.save(encoder.state_dict(),
                       f"model_weights_encoder_{i}.pth")  # saves everything from the state dictionary
            torch.save(decoder.state_dict(),
                       f"model_weights_decoder_{i}.pth")  # saves everything from the state dictionary
        beg = time.time()
        crumpled, smooth = train_sampler.next()
        crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
        smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
        embedding, activations = encoder.forward(crumpled) # sanity check: can we recreate? #TODO: this should be crumpled
        predicted_smooth = decoder(embedding, None)
        encoding_loss = loss(smooth, predicted_smooth) #+ norm_mult * torch.sum(torch.abs(out))

        if i % 25 == 0:
            print(i, " ", encoding_loss.cpu().detach().numpy())

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoding_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        csv_train_writer.writerow([i, encoding_loss.cpu().detach().item()])
        writer.add_scalar("Loss/train_loss", encoding_loss, i)
        # print(time.time() - beg)
    writer.close()
