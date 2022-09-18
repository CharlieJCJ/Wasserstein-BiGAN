from stylegan2 import Generator, Discriminator
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from constants import *
from util import JointCritic, WALI, ResNetSimCLR, ContrastiveLearningDataset, DeterministicConditional


def create_generator():
  return Generator(IMAGE_SIZE, H_DIM, 8)

def create_critic():
  x_mapping = Discriminator(IMAGE_SIZE)
  # FIXME -  Question: what is DIM?
  # kernal size is 1, stride is 1, padding is 0. 
  z_mapping = nn.Sequential(
    Conv2d(H_DIM, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))
  
  # DIM_D = 8192 - dimension from discriminator
  joint_mapping = nn.Sequential(
    Conv2d(DIM_D + 512, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 1, 1, 1, 0))
  # last output channel is 1, output 0, 1.

  return JointCritic(x_mapping, z_mapping, joint_mapping)


def create_WALI():
  E = ResNetSimCLR(H_DIM, Z_DIM)
  G = create_generator()
  C = create_critic()
  wali = WALI(E, G, C)
  return wali