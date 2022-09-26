#-> Dataset specify
dataset = "mnist"

#-> Baseline setting
# baseline = True # -- click

#-> Multiple GPU setting
CUDA_VISIBLE_DEVICES = '0, 1, 2, 3'
# GPUS = 4 # -- click
# N_VIEW = 2 # -- click default 2

#-> Hyperparameters (included in click)
# BATCH_SIZE = 16 * 4  # Original = 256, we start with something smaller
# ITER = 200000 # Number of epochs to train for
# H_DIM = 512
# Z_DIM = 128
# NLAT = 512 # same as H_DIM
# LEAK = 0.2
# C_ITERS = 1       # critic iterations
# EG_ITERS = 1      # encoder / generator iterations
# LAMBDA = 10       # strength of gradient penalty
# LEARNING_RATE = 1e-5
# BETA1 = 0.5
# BETA2 = 0.9
# DIM = 128         # Network dimension
#-> Writer settings
WRITER_ITER = 10 # log every 10 iterations

if dataset == "cifar10":
    # training hyperparameters
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    DIM_D = 8192 # Need to check the size in stylegan2.py using test(); checked
    
    # dataset path
    CIFAR_PATH = r'~\torch\data\CIFAR10'


