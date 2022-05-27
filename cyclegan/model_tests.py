from cyclegan.models import Discriminator, Generator, Composite
from torchinfo import summary

shape = (1, 3, 128, 128)
summary(Discriminator(), shape)
summary(Generator(), shape)
summary(Composite(Generator(), Discriminator(), Generator()), [shape, shape])

