import pickle
from torch import nn
import time
import matplotlib.pyplot as plt
# from EncoderDecoder import Encoder, Decoder, Discriminator

from cyclegan.models import Generator, Discriminator
import csv
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from torchsummary import summary

sampler_dataset = pickle.load(open("frozen_datasets/dataset_49000_small.pkl", "rb"))
sampler_dataset.set_mode("single_sample")

print("done loading data")

# TODO: add csv logging on top of tensorboard because it's not working

from torch.utils.data import DataLoader
from torch.utils.data import random_split

valid_size = 128
valid, train = random_split(sampler_dataset, [valid_size, 49000 - valid_size])
train_generator = DataLoader(train, batch_size=8, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)

ax_objects = generate_plot(4) #creates plots that help with visualization

def make_models():
    gen_c_to_uc = Generator(input_shape=(3, 128, 128))
    disc_uc = Discriminator(input_shape=(6, 128, 128))
    return gen_c_to_uc , disc_uc


def test_evaluate(generator, device, step, save = True):
    loss = nn.MSELoss()
    random_selection = np.random.randint(valid_size)
    loss_value = 0
    MI_value = 0
    MI_base = 0
    MI_low = 0
    valid_sampler = iter(valid_generator)
    generator.train(False)
    for i in range(valid_size):
        with torch.no_grad():
            crumpled, smooth = valid_sampler.next()
            crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
            smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
            proposed_smooth = generator(crumpled)
            loss_value += loss(smooth, proposed_smooth)

            hist, value = generate_mutual_information(to_numpy(smooth), to_numpy(proposed_smooth), hist = True)
            hist_log = np.zeros(hist.shape)
            non_zeros = hist != 0
            hist_log[non_zeros] = np.log(hist[non_zeros])

            MI_value += value
            MI_base += generate_mutual_information(to_numpy(smooth), to_numpy(smooth))
            MI_low += generate_mutual_information(to_numpy(smooth), to_numpy(crumpled))
            if i == random_selection:
                visualize(ax_objects, [to_numpy(crumpled[0]), to_numpy(smooth[0]), to_numpy(proposed_smooth[0]), hist_log],
                          ["crumpled", "smooth", "output", "Mutual Info"], save = save, step = step, visible = True)

    writer.add_scalar("Loss/valid", loss_value, step)
    writer.add_scalar("Loss/valid_MI", MI_value, step)
    print(f"Mutual information value (higher better): {MI_value}, which is upper bounded by {MI_base} and lower bounded by {MI_low}")
    print(f"validation loss: {loss_value.item()} (for scale: {loss_value.item() / (valid_size)}")
    csv_valid_writer.writerow([step, MI_value, loss_value.item()])
    generator.train(True)

def discriminator_loss(logits_real, logits_fake, device):
    bce = nn.BCEWithLogitsLoss()
    true_labels = torch.ones((logits_real.shape[0],)).to(device)
    false_labels = torch.zeros((logits_fake.shape[0],)).to(device)
    real_score = bce(torch.squeeze(logits_real), true_labels)
    fake_score = bce(torch.squeeze(logits_fake), false_labels)
    loss = real_score + fake_score
    loss = loss / 2 # slows down learning, as specified in the paper
    return loss

def generator_loss(logits_fake, device):
    bce = nn.BCEWithLogitsLoss()
    true_labels = torch.ones((logits_fake.shape[0],)).to(device)
    real_score = bce(torch.squeeze(logits_fake), true_labels)
    return real_score

if __name__ == "__main__":
    experiment = "GAN_paired_loss"
    load_model = False

    num_training_steps = 50000
    path = f"G:\\Desktop\\Working Repository\\CRUMPL\\experiments\\{experiment}"

    writer = SummaryWriter(path)  # you can specify logging directory

    generator, discriminator = make_models()


    print("done generating and loading models")
    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    else:
        device = "cpu"

    # if load_model:
    #     checkpoint = 100000
    #     encoder.load_state_dict(torch.load(f'{path}/model_weights_encoder_{checkpoint}.pth'))
    #     decoder.load_state_dict(torch.load(f'{path}/model_weights_decoder_{checkpoint}.pth'))
    #     test_evaluate(encoder, decoder, device, step = "TEST", save = True)
    #     quit()

    torch.autograd.set_detect_anomaly(True)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5,0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5,0.999))

    soft_make_dir(path)
    os.chdir(path)
    f = open("metrics_train.csv", "w", newline="")
    csv_train_writer = csv.writer(f)
    csv_train_writer.writerow(["step", "generator", "discriminator"])
    f = open("metrics_valid.csv", "w", newline="")
    csv_valid_writer = csv.writer(f)
    csv_valid_writer.writerow(["step", "MI", "MSE"])

    norm_mult = 1e-7
    train_sampler = iter(train_generator)
    loss = nn.L1Loss()
    for i in range(num_training_steps + 1):
        if i % 1527 == 0:
            train_sampler = iter(train_generator)

        if i % 50 == 0:
            writer.flush()
            print("eval time!")
            test_evaluate(generator, device, step = i, save = True)
        if i % 2500 == 0:
            torch.save(generator.state_dict(),
                       f"model_weights_generator_{i}.pth")  # saves everything from the state dictionary
            torch.save(discriminator.state_dict(),
                       f"model_weights_discriminator_{i}.pth")  # saves everything from the state dictionary

        crumpled, smooth = train_sampler.next()
        crumpled = to_tensor(crumpled, device)
        smooth = to_tensor(smooth, device)

        # discriminator step
        discriminator_optimizer.zero_grad()
        predicted_smooth = generator(crumpled).detach()
        real_logits = discriminator(torch.cat([smooth, crumpled], dim = 1))
        fake_logits = discriminator(torch.cat([predicted_smooth, crumpled], dim = 1))
        d_loss = discriminator_loss(real_logits, fake_logits, device)
        d_loss.backward()
        discriminator_optimizer.step()

        # generator step
        generator_optimizer.zero_grad()
        predicted_smooth = generator(crumpled)
        fake_logits = discriminator(torch.cat([predicted_smooth, crumpled], dim = 1))
        g_loss = generator_loss(fake_logits, device) + 100 * loss(smooth, predicted_smooth)
        # g_loss = loss(smooth, predicted_smooth)
        g_loss.backward()
        generator_optimizer.step()

        if i % 1 == 0:
            print(i, " Discriminator ", d_loss.cpu().detach().item())
            print(i, " Generator ", g_loss.cpu().detach().item())

        csv_train_writer.writerow([i, g_loss.cpu().detach().item(), d_loss.cpu().detach().item()])
        writer.add_scalar("Loss/generator_loss", g_loss, i)
        writer.add_scalar("Loss/discriminator_loss", d_loss, i)
        # print(time.time() - beg)
    writer.close()
