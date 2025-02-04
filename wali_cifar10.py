from asyncio import constants
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from util import JointCritic, WALI, ResNetSimCLR, ContrastiveLearningDataset, DeterministicConditional
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import logging
from constants import *
from models import create_WALI
import click
cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)




# Training pipeline function
@click.command()
@click.option('--model', type=str, help='Model filename', required=True)
@click.option('--log', type=str, help='logName', required=True)
def train(model, log):
  logging.basicConfig(filename=f'{log}.log', level=logging.DEBUG)
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
  torch.save(wali.state_dict(), f'cifar10/models/{model} init.ckpt')
  for curr_iter in range(ITER):
    for batch_idx, (x, _) in enumerate(train_loader, 1):
      running_losses = [0, 0]
      ############################
      # Starting BiGAN procedures
    
      # Curr_iter starts from one, and stores to init_x
      # if curr_iter == 0:
      #   init_x = original_imgs # init_x is used to plot later (not very important in training)
      #   curr_iter += 1
      
      # Forward pass, get loss
      # Sample z from a prior distribution ~ N(0, 1)
      # original_imgs.size(0) = batch size
      # x[2] is the original image TODO
      z = torch.randn(x[2].size(0), H_DIM, 1, 1).to(device)
      C_loss, EG_loss = wali(x, z, lamb=LAMBDA, device=device, baseline = baseline)
      running_losses[0] += C_loss.item()
      running_losses[1] += EG_loss.item()
      # print("loss calculated C_loss: ", C_loss, "EG_loss: ",  EG_loss)
      if batch_idx % WRITER_ITER == 0:
        print('Epoch: {}, Batch: {} C_loss: {:.4f}, EG_loss: {:.4f}'.format(
          curr_iter, batch_idx, C_loss.item(), EG_loss.item()))
        logging.info('C_loss: ' + str(running_losses[0]) + 'EG_loss: '+ str(running_losses[1]) + " epoch: " + str(curr_iter) + " batch"+ str((curr_iter) * n_total_runs + batch_idx))
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
    if curr_iter % 5 == 0:
      torch.save(wali.state_dict(), f'cifar10/models/{model} epoch {curr_iter}.ckpt')
      print(f'Model saved to cifar10/models/{model} epoch {curr_iter}.ckpt')
      logging.info(f"Model saved to cifar10/models/{model} epoch {curr_iter}.ckpt")
    
    # Outside of batch for loop (simclr schedule updates)
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
  train()