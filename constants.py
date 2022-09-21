
# training hyperparameters
N_VIEW = 2
BATCH_SIZE = 16 * 4  # Original = 256, we start with something smaller
ITER = 200000 # Number of epochs to train for
IMAGE_SIZE = 32
NUM_CHANNELS = 3
H_DIM = 512
Z_DIM = 128
NLAT = 512
DIM_D = 8192 # Need to check the size in stylegan2.py using test(); checked

LEAK = 0.2

DIM = 128
C_ITERS = 1       # critic iterations
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1e-5
BETA1 = 0.5
BETA2 = 0.9

# dataset path
CIFAR_PATH = r'~\torch\data\CIFAR10'

WRITER_ITER = 10

# Baseline setting
baseline = True

CUDA_VISIBLE_DEVICES = '0, 1, 6, 7'
GPUS = 4