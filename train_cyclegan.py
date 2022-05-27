import torch

from pipeline_whole import CrumpleLibrary
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from cyclegan.models import Generator, Discriminator, Composite

INPUT_SHAPE = (3, 128, 128)

generated_library = CrumpleLibrary(base_directory="../data/val_blurred/image_exports/", number_images=100)
generated_library.set_mode('single_sample')

valid_size = 5
batch_size = 9
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


def train_cyclegan(n_epochs=1):
    gen_c_to_uc = Generator(input_shape=INPUT_SHAPE)
    gen_uc_to_c = Generator(input_shape=INPUT_SHAPE)

    disc_c = Discriminator(input_shape=INPUT_SHAPE)
    disc_uc = Discriminator(input_shape=INPUT_SHAPE)

    comp_c_to_uc = Composite(gen_c_to_uc, disc_uc, gen_uc_to_c)
    comp_uc_to_c = Composite(gen_uc_to_c, disc_c, gen_c_to_uc)

    c_to_uc_optim = optim.Adam(comp_c_to_uc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    uc_to_c_optim = optim.Adam(comp_uc_to_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_c_optim = optim.Adam(disc_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_uc_optim = optim.Adam(disc_uc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        train_sampler = iter(train_generator)
        for i, (x_real_c, x_real_uc) in enumerate(train_sampler):
            x_real_c = x_real_c.type(torch.FloatTensor)
            x_real_uc = x_real_uc.type(torch.FloatTensor)
            print(type(x_real_c), type(x_real_uc))
            # x_real_c = torch.from_numpy()

            y_real_c = torch.ones((x_real_c.shape[0]))
            y_real_uc = torch.ones((x_real_uc.shape[0]))

            x_fake_c = generate_fake_samples(gen_uc_to_c, x_real_uc)
            x_fake_uc = generate_fake_samples(gen_c_to_uc, x_real_c)

            y_fake_c = torch.zeros((x_real_c.shape[0]))
            y_fake_uc = torch.zeros((x_real_uc.shape[0]))

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
    train_cyclegan()
