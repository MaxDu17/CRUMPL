import torch
import pickle
import csv

from pipeline_whole import CrumpleLibrary
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from cyclegan.models import Composite
from EncoderDecoder import PixGenerator, PixDiscriminator

from utils.utils import *

INPUT_SHAPE = (3, 128, 128)

generated_library = pickle.load(open("frozen_datasets/dataset_49000_small.pkl", "rb"))
generated_library.set_mode('single_unpaired_sample')
print("Done loading data")

ax_objects = generate_plot(5) #creates plots that help with visualization

valid_size = 128
batch_size = 8
valid, train = random_split(generated_library, [valid_size, len(generated_library) - valid_size])
train_generator = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
valid_generator = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0)

d_loss = nn.MSELoss()
id_loss = nn.L1Loss()
f_loss = nn.L1Loss()
b_loss = nn.L1Loss()

def train_generator_model(generator, optimizer, x, y):
    optimizer.zero_grad()
    output_d, output_id, output_f, output_b = generator(*x)
    # output_d is the discrimnator output
    # output_id is the identity operator output
    # output_f is forward cycle
    # output_b is backward cycle

    loss = d_loss(output_d, y[0]) \
           + 5 * id_loss(output_id, y[1]) \
           + 10 * f_loss(output_f, y[2]) \
           + 10 * b_loss(output_b, y[3])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_discriminator_model(discriminator, optimizer, x, y):
    optimizer.zero_grad()
    output_d = discriminator(x)
    loss = 0.5 * d_loss(output_d, y)
    loss.backward()
    optimizer.step()

    return loss.item()


def generate_fake_samples(gen, x):
    with torch.no_grad():
        return gen(x)


def test_evaluate(uncrumpler, crumpler, device, step, writer = None, csv_writer = None, save = True):
    loss = nn.MSELoss()
    random_selection = np.random.randint(valid_size)
    loss_value = 0
    MI_value = 0
    MI_base = 0
    MI_low = 0
    generated_library.set_mode('single_sample')
    valid_sampler = iter(valid_generator)
    crumpler.train(False)
    uncrumpler.train(False)
    for i in range(valid_size):
        with torch.no_grad():
            crumpled, smooth = valid_sampler.next()
            crumpled = torch.as_tensor(crumpled, device=device, dtype = torch.float32)
            smooth = torch.as_tensor(smooth, device = device, dtype = torch.float32)
            proposed_smooth = uncrumpler(crumpled)
            recrumpled = crumpler(smooth)
            loss_value += loss(smooth, proposed_smooth)

            hist, value = generate_mutual_information(to_numpy(smooth), to_numpy(proposed_smooth), hist = True)
            hist_log = np.zeros(hist.shape)
            non_zeros = hist != 0
            hist_log[non_zeros] = np.log(hist[non_zeros])

            MI_value += value
            MI_base += generate_mutual_information(to_numpy(smooth), to_numpy(smooth))
            MI_low += generate_mutual_information(to_numpy(smooth), to_numpy(crumpled))
            if i == random_selection:
                visualize(ax_objects, [to_numpy(crumpled[0]), to_numpy(smooth[0]), to_numpy(recrumpled[0]), to_numpy(proposed_smooth[0]), hist_log],
                          ["crumpled", "smooth", "output", "recrumpled", "Mutual Info"], save = save, step = step, visible = True)

    if writer is not None:
        writer.add_scalar("Loss/valid", loss_value, step)
        writer.add_scalar("Loss/valid_MI", MI_value, step)
    if csv_writer is not None:
        csv_writer.writerow([step, MI_value, loss_value.item()])

    print(f"Mutual information value (higher better): {MI_value}, which is upper bounded by {MI_base} and lower bounded by {MI_low}")
    print(f"validation loss: {loss_value.item()} (for scale: {loss_value.item() / (valid_size)}")
    uncrumpler.train(True)
    crumpler.train(True)
    generated_library.set_mode('single_unpaired_sample')

def train_cyclegan(n_epochs=1):
    gen_c_to_uc = PixGenerator(INPUT_SHAPE)
    gen_uc_to_c = PixGenerator(INPUT_SHAPE)

    disc_c = PixDiscriminator(INPUT_SHAPE)
    disc_uc = PixDiscriminator(INPUT_SHAPE)

    comp_c_to_uc = Composite(gen_c_to_uc, disc_uc, gen_uc_to_c)
    comp_uc_to_c = Composite(gen_uc_to_c, disc_c, gen_c_to_uc)

    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
        comp_c_to_uc.cuda()
        comp_uc_to_c.cuda()
    else:
        device = "cpu"

    c_to_uc_optim = optim.Adam(comp_c_to_uc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    uc_to_c_optim = optim.Adam(comp_uc_to_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_c_optim = optim.Adam(disc_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_uc_optim = optim.Adam(disc_uc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        train_sampler = iter(train_generator)
        for i, (x_real_c, x_real_uc) in enumerate(train_sampler): #todo: replace with random sampling
            if i % 100 == 0:
                test_evaluate(gen_c_to_uc, gen_uc_to_c, device, step = f"{epoch}_{i}", writer = writer, csv_writer = csv_valid_writer)

            if i % 2500 == 0:
                print("saving!")
                torch.save(comp_c_to_uc.state_dict(),
                           f"comp_c_to_uc_{epoch}_{i}.pth")  # saves everything from the state dictionary
                torch.save(comp_uc_to_c.state_dict(),
                           f"comp_uc_to_c_{epoch}_{i}.pth")  # saves everything from the state dictionary

            x_real_c = to_tensor(x_real_c, device)
            x_real_uc = to_tensor(x_real_uc, device)

            y_real_c = torch.ones((x_real_c.shape[0], 1, 4, 4)).to(device)
            y_real_uc = torch.ones((x_real_uc.shape[0], 1, 4, 4)).to(device)

            x_fake_c = generate_fake_samples(gen_uc_to_c, x_real_uc)
            x_fake_uc = generate_fake_samples(gen_c_to_uc, x_real_c)

            y_fake_c = torch.zeros((x_real_c.shape[0], 1, 4, 4)).to(device)
            y_fake_uc = torch.zeros((x_real_uc.shape[0], 1, 4, 4)).to(device)

            # update uncrumpled -> crumpled generator
            uc_to_c_loss = train_generator_model(comp_uc_to_c, uc_to_c_optim, [x_real_uc, x_real_c],
                                                 [y_real_c, x_real_c, x_real_uc, x_real_c])

            # update discriminator for crumpled -> [real/fake]
            disc_c_loss1 = train_discriminator_model(disc_c, disc_c_optim, x_real_c, y_real_c)
            disc_c_loss2 = train_discriminator_model(disc_c, disc_c_optim, x_fake_c, y_fake_c)

            # update crumpled -> uncrumpled generator
            c_to_uc_loss = train_generator_model(comp_c_to_uc, c_to_uc_optim, [x_real_c, x_real_uc],
                                                 [y_real_uc, x_real_uc, x_real_c, x_real_uc])

            # update discriminator for uncrumpled -> [real/fake]
            disc_uc_loss1 = train_discriminator_model(disc_uc, disc_uc_optim, x_real_uc, y_real_uc)
            disc_uc_loss2 = train_discriminator_model(disc_uc, disc_uc_optim, x_fake_uc, y_fake_uc)

            print(f"Epoch {epoch + 1}, batch {i}: disc_c[{disc_c_loss1: .3f}, {disc_c_loss2: .3f}]; disc_uc[{disc_uc_loss1: .3f}, {disc_uc_loss2: .3f}]; uc_to_c[{uc_to_c_loss: .3f}]; c_to_uc[{c_to_uc_loss: .3f}]")


if __name__ == '__main__':
    experiment = "U_Net_cyclegan"
    load_model = False

    path = f"G:\\Desktop\\Working Repository\\CRUMPL\\experiments\\{experiment}"
    soft_make_dir(path)
    os.chdir(path)

    if torch.cuda.is_available():
        print("cuda available!")
        device = "cuda"
    else:
        device = "cpu"

    if load_model:
        checkpoint = "4_5000"
        gen_c_to_uc = Generator(input_shape=INPUT_SHAPE)
        gen_uc_to_c = Generator(input_shape=INPUT_SHAPE)
        disc_c = Discriminator(input_shape=INPUT_SHAPE)
        disc_uc = Discriminator(input_shape=INPUT_SHAPE)

        comp_c_to_uc = Composite(gen_c_to_uc, disc_uc, gen_uc_to_c).to(device)
        comp_uc_to_c = Composite(gen_uc_to_c, disc_c, gen_c_to_uc).to(device)
        comp_c_to_uc.load_state_dict(torch.load(f'{path}/comp_c_to_uc_{checkpoint}.pth'))
        comp_uc_to_c.load_state_dict(torch.load(f'{path}/comp_c_to_uc_{checkpoint}.pth'))
        test_evaluate(gen_c_to_uc, gen_uc_to_c, device = "cuda", step="TEST")
        run_through_model(gen_c_to_uc, "../../data/crumple_test/", f"{path}/arbitrary_eval/", 128, device)
        quit()

    num_epochs = 5


    f = open("metrics_valid.csv", "w", newline="")
    csv_valid_writer = csv.writer(f)
    csv_valid_writer.writerow(["step", "MI", "MSE"])
    writer = SummaryWriter(path)  # you can specify logging directory

    train_cyclegan(num_epochs) #5 epochs is around 30k steps
