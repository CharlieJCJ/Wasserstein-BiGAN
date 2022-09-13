import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from util import JointCritic, WALI, ResNetSimCLR, ContrastiveLearningDataset, DeterministicConditional
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from stylegan2 import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
import logging

WRITER_ITER = 10
cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


# training hyperparameters
N_VIEW = 2
BATCH_SIZE = 16 # Original = 256, we start with something smaller
ITER = 200000 # Number of epochs to train for
IMAGE_SIZE = 32
NUM_CHANNELS = 3
H_DIM = 512
Z_DIM = 128
NLAT = 512
DIM_D = 8192 # Need to check the size in stylegan2.py using test()

LEAK = 0.2

# FIXME
DIM = 128
C_ITERS = 5       # critic iterations
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.9

# dataset path
CIFAR_PATH = r'~\torch\data\CIFAR10'

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

# Training pipeline function
def main():
  logging.basicConfig(filename='run1.log', level=logging.DEBUG)
  logging.info('Start training')
  writer = SummaryWriter("runs/cifar10")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Device:', device)
  wali = create_WALI().to(device)

  # Load CIFAR10 dataset
  dataset = ContrastiveLearningDataset("./datasets")
  # Each have 2 views (2 views + original image)
  train_dataset = dataset.get_dataset("cifar10", N_VIEW)

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=BATCH_SIZE, shuffle=True,
      num_workers=12, pin_memory=True, drop_last=True)
  n_total_runs = len(train_loader)
  # FIXME - wali.get_encoder_parameters() might be the entire resnet + MLP. - FIXED
  optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
  optimizerC = Adam(wali.get_critic_parameters(), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
  # SimCLR Encoder and training scheduler
  # optimizerSimCLR = torch.optim.Adam(wali.get_encoder_parameters(), 0.0003, weight_decay=1e-4)

  # schedulerSimCLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerSimCLR, T_max=len(train_loader), eta_min=0,
  #                                                          last_epoch=-1)
  # scalerSimCLR = GradScaler(enabled=True)
  # criterionSimCLR = torch.nn.CrossEntropyLoss().to(device)
  noise = torch.randn(64, NLAT, 1, 1, device=device)
  
  # Debugging purposes :down
  # test_size(train_loader)
  EG_losses, C_losses, R_losses, Constrastive_losses = [], [], [], []
  curr_iter = C_iter = EG_iter = 0
  C_update, EG_update = True, False
  print('Training starts...')
  
  while curr_iter < ITER:
    for batch_idx, (x, _) in enumerate(train_loader, 1):
      running_losses = [0, 0]
      print("batch_idx: ", batch_idx)

      # Transformed, original image
      transformed_imgs = torch.cat([x[0], x[1]], dim=0) # expecting 512 * 3 * 32 * 32 (batch size is 256)
      original_imgs = x[2]
      transformed_imgs = transformed_imgs.to(device)
      original_imgs = original_imgs.to(device)
      # print("data loaded")
      # print(original_imgs.shape)
      ############################
      # Starting BiGAN procedures
    
      # Curr_iter starts from one, and stores to init_x
      if curr_iter == 0:
        init_x = original_imgs # init_x is used to plot later (not very important in training)
        curr_iter += 1
      
      # Forward pass, get loss
      # Sample z from a prior distribution ~ N(0, 1)
      # original_imgs.size(0) = batch size
      # x[2] is the original image TODO
      z = torch.randn(x[2].size(0), H_DIM, 1, 1).to(device)
      C_loss, EG_loss = wali(x, z, lamb=LAMBDA, device=device)
      running_losses[0] += C_loss.item()
      running_losses[1] += EG_loss.item()
      # print("loss calculated C_loss: ", C_loss, "EG_loss: ",  EG_loss)
      if batch_idx % WRITER_ITER == 0:
        print('Iter: {}, Batch: {} C_loss: {:.4f}, EG_loss: {:.4f}'.format(
          curr_iter, batch_idx, C_loss.item(), EG_loss.item()))
        # writer.add_scalar('C_loss', running_losses[0], (curr_iter - 1) * n_total_runs + batch_idx)
        # writer.add_scalar('EG_loss', running_losses[1], (curr_iter - 1) * n_total_runs + batch_idx)
        logging.info('C_loss: ' + str(running_losses[0]) + 'EG_loss: '+ str(running_losses[1]) + " epoch: " + str(curr_iter - 1) + " batch"+ str((curr_iter - 1) * n_total_runs + batch_idx))
      # C_update: C_loss and Reconstruction loss
      if C_update:
        print("C_update")
        optimizerC.zero_grad()
        C_loss.backward()
        # C_losses.append(C_loss.item())
        optimizerC.step()

        C_iter += 1
        
        # Switch C to EG update
        if C_iter == C_ITERS:
          C_iter = 0
          C_update, EG_update = False, True
        continue

      # EG_update: EG_loss and SimCLR loss (contrastive loss), EG loss contains SimCLR loss already in forward pass 
      if EG_update:
        print("EG_update")
        optimizerEG.zero_grad()
        EG_loss.backward()
        # EG_losses.append(EG_loss.item())
        optimizerEG.step()
        EG_iter += 1
        if EG_iter == EG_ITERS:
          EG_iter = 0
          C_update, EG_update = True, False
          curr_iter += 1
        else:
          continue
      
        
      # # print training statistics
      # if curr_iter % 100 == 0:
      #   print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
      #     % (curr_iter, ITER, EG_loss.item(), C_loss.item()))

      #   # plot reconstructed images and samples
      #   wali.eval()
      #   real_x, rect_x = init_x[:32], wali.reconstruct(init_x[:32]).detach_()
      #   rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1) 
      #   rect_imgs = rect_imgs.view(64, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
      #   genr_imgs = wali.generate(noise).detach_().cpu()
      #   utils.save_image(rect_imgs * 0.5 + 0.5, 'cifar10/rect%d.png' % curr_iter)
      #   utils.save_image(genr_imgs * 0.5 + 0.5, 'cifar10/genr%d.png' % curr_iter)
      #   wali.train()

      # save model
      if curr_iter % 10 == 0:
        torch.save(wali.state_dict(), 'cifar10/models/%d.ckpt' % curr_iter)
    
    # Outside of batch for loop ( simclr schedule updates)
    # if curr_iter >= 10:
    #     schedulerSimCLR.step()
  print("End of training")
  # # plot training loss curve
  # plt.figure(figsize=(10, 5))
  # plt.title('Training loss curve')
  # plt.plot(EG_losses, label='Encoder + Generator')
  # plt.plot(C_losses, label='Critic')
  # plt.xlabel('Iterations')
  # plt.ylabel('Loss')
  # plt.legend()
  # plt.savefig('cifar10/loss_curve.png')

def test_size(train_loader):
  for batch_idx, (x, _) in enumerate(train_loader, 1):
    # x[0] is the first view of the first batch
    # x[1] is the second view of the first batch
    # x[2] is the original iamges of the first batch
    print("batch_idx: ", batch_idx)
    print("x: ", x[0], x[1], x[2])
    print("x: ", x[0].shape, x[1].shape, x[2].shape)
    break


if __name__ == "__main__":
  main()