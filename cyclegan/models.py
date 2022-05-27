import torch.nn as nn

# constants used as default values for input shapes
import torch.nn.init

INPUT_WIDTH = 128
INPUT_HEIGHT = 128
INPUT_CHANNELS = 3


def init_conv_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0, std=0.02)


class Discriminator(nn.Module):
    def __init__(self, input_shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        ])

        channels = (128, 256, 512)
        for ch in channels:
            self.layers.append(nn.Conv2d(ch // 2, ch, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.InstanceNorm2d(num_features=ch))
            self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.layers.extend([
            nn.Conv2d(512, 512, kernel_size=4, padding='same'),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            # output
            # nn.Conv2d(512, 1, kernel_size=4, padding='same')
            nn.Conv2d(512, 1, kernel_size=input_shape[1] // 16)
        ])

        self.layers.apply(init_conv_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, input_filters, output_filters):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_filters, output_filters, kernel_size=3, padding='same'),
            nn.InstanceNorm2d(output_filters),
            nn.ReLU(),
            nn.Conv2d(output_filters, output_filters, kernel_size=3, padding='same'),
            nn.InstanceNorm2d(output_filters),
        )
        self.layers.apply(init_conv_weights)

    def forward(self, x):
        return torch.cat((x, self.layers(x)), dim=1)


class Generator(nn.Module):
    def __init__(self, n_resnet=9, input_shape=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(input_shape[0], 64, kernel_size=7, padding='same'),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU()
        ])

        for i in range(1, n_resnet + 1):
            self.layers.append(ResNetBlock(input_filters=i * 256, output_filters=256))

        self.layers.extend([
            nn.ConvTranspose2d(256 * (n_resnet + 1), 128, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, input_shape[0], kernel_size=7, padding='same'),
            nn.InstanceNorm2d(num_features=input_shape[0]),
            nn.Tanh()
        ])

        self.layers.apply(init_conv_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Composite(nn.Module):
    """
    Converts from class S to class T
    """

    def __init__(self, gen_forward, disc_forward, gen_backward):
        super().__init__()
        self.gen_forward = gen_forward  # converts S to T
        self.disc = disc_forward
        self.gen_backward = gen_backward  # converts S to T

    def forward(self, image_s, image_t):
        gen_forward_out = self.gen_forward(image_s)
        with torch.no_grad():
            output_d = self.disc(gen_forward_out)

        # since gen_forward converts from S to T, putting in T should be identity
        output_id = self.gen_forward(image_t)

        # forward cycle
        with torch.no_grad():
            output_f = self.gen_backward(gen_forward_out)

        # backward cycle
        with torch.no_grad():
            gen_backward_out = self.gen_backward(image_t)
        output_b = self.gen_forward(gen_backward_out)

        return output_d, output_id, output_f, output_b
