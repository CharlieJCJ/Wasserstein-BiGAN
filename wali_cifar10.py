from email.mime import base
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
import click
# from constants import *
import os
import datetime
datetime_object = datetime.datetime.now()
print(datetime_object)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3, 4, 5"
WRITER_ITER = 10
MODELSAVE_ITER = 2000 # save every 5000 batches
# cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.empty_cache()
SAMPLESAVE_ITER = 50
# cudnn.deterministic = True


# Training pipeline function
@click.command()
@click.option('--model', type=str, help='Model filename', required=True)
@click.option('--log', type=str, help='logName', required=True)
@click.option('--baseline', type=bool, help='baseline', default= False, show_default=True)
@click.option('--N_VIEW', type=int, help='Number of views', default=2, show_default=True)
@click.option('--BATCH_SIZE', type=int, help='Batch size', default=16, show_default=True)
@click.option('--ITER', type=int, help='Number of epochs to train for', default=200000, show_default=True)
@click.option('--H_DIM', type=int, help='Hidden dimension', default=512, show_default=True)
@click.option('--Z_DIM', type=int, help='Latent dimension', default=128, show_default=True)
@click.option('--NLAT', type=int, help='NLAT', default=512, show_default=True)
@click.option('--LEAK', type=float, help='Leak', default=0.2, show_default=True)
@click.option('--C_ITERS', type=int, help='Critic iterations', default=1, show_default=True) # Citer = 5
@click.option('--EG_ITERS', type=int, help='Encoder / generator iterations', default=1)
@click.option('--LAMBDAS', type=int, help='Strength of gradient penalty', default=10)
@click.option('--LEARNING_RATE', type=float, help='Learning rate', default=0.002, show_default=True)
@click.option('--BETA1', type=float, help='BETA1', default=0.0)
@click.option('--BETA2', type=float, help='BETA2', default=0.999)
@click.option('--VISUAL_NUM', type=int, help='VISUAL_NUM', default=8)
@click.option('--DATASET', help='Dataset name', type=click.Choice(['cifar10', 'mnist', 'celeba', 'LSUN']), default='cifar10', show_default=True)
@click.option('--CUDA_VISIBLE_DEVICES', help='CUDA_VISIBLE_DEVICES', type=str, default='0', show_default=True)
@click.option('--LOAD', help='Load Path', type=str, default="", show_default=True)
# @click.option('--LOCAL', help='Local Computer with small GPU memory', type=bool, default=False, show_default=True)
def main(model, 
         log, 
         baseline, 
         n_view, 
         batch_size, 
         iter, 
         h_dim, 
         z_dim, 
         nlat, 
         leak,
         c_iters, 
         eg_iters, 
         lambdas, 
         learning_rate, 
         beta1, 
         beta2, 
         visual_num, 
         dataset, 
         cuda_visible_devices,
         load):
  MODEL,LOG,BASELINE, N_VIEW, BATCH_SIZE, ITER,  H_DIM, Z_DIM, NLAT, LEAK,C_ITERS, EG_ITERS, LAMBDAS, LEARNING_RATE, BETA1, BETA2, VISUAL_NUM, DATASET, CUDA_VISIBLE_DEVICES, LOAD = model,log,baseline, n_view, batch_size, iter, h_dim, z_dim,nlat,leak,c_iters,eg_iters,lambdas,learning_rate,beta1,beta2,visual_num,dataset,cuda_visible_devices, load
  # os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
  originalBATCH = BATCH_SIZE
  traindir = f"train/{DATASET}-{datetime_object}"
  modeldir = f"{traindir}/models"
  os.makedirs(traindir, exist_ok=True)
  os.makedirs(modeldir, exist_ok=True)

  
  print("GPUs: ", torch.cuda.device_count())
  Parallel_Index = [int(item) for item in CUDA_VISIBLE_DEVICES.split(',') if item.isdigit()]
  GPUS = len(Parallel_Index)
  print("GPUs used in training: ", GPUS)
  BATCH_SIZE = BATCH_SIZE * GPUS# batch size for each GPU, total batch size is BATCH_SIZE * GPUS
  # if LOCAL:
  #   BATCH_SIZE = 2
  # Extra dataset dependent hyperparameters
  if DATASET == 'cifar10':
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    DIM_D = 8192        # Need to check the size in stylegan2.py using test(); checked
  elif DATASET == 'mnist':
    print("Using MNIST")
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    DIM_D = 8192 
  elif DATASET == 'LSUN':
    IMAGE_SIZE = 128
    NUM_CHANNELS = 3
    DIM_D = 8192
    MODELSAVE_ITER = 2000 # LSUN is huge dataset (3M images)
  else:   # default
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    DIM_D = 8192        # Need to check the size in stylegan2.py using test(); checked
  logging.basicConfig(filename=f'{traindir}/{LOG}.log', level=logging.DEBUG)
  logging.info('Start training')
  # writer = SummaryWriter("runs/cifar10")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Device:', device)
  wali = create_WALI(H_DIM, Z_DIM, LEAK, DIM_D, IMAGE_SIZE).to(device)
  
  if LOAD != "":
    print("Loading model from ", LOAD)
    wali.load_state_dict(torch.load(LOAD))

  # Load CIFAR10 dataset
  dataset = ContrastiveLearningDataset("./datasets")
  # Each have 2 views (2 views + original image)
  train_dataset = dataset.get_dataset(DATASET, N_VIEW)

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=BATCH_SIZE, shuffle=True,
      num_workers=1, drop_last=True)
  n_total_runs = len(train_loader)
  print('Number of runs:', n_total_runs)
  # FIXME - wali.get_encoder_parameters() might be the entire resnet + MLP. - FIXED
  optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=2.5e-5)
  optimizerC = Adam(wali.get_critic_parameters(), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2), weight_decay=2.5e-5)
  # SimCLR Encoder and training scheduler
  # optimizerSimCLR = torch.optim.Adam(wali.get_encoder_parameters(), 0.0003, weight_decay=1e-4)

  # schedulerEG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerEG, T_max=len(train_loader), eta_min=0,
  #                                                          last_epoch=-1)
  # scalerSimCLR = GradScaler(enabled=True)
  # criterionSimCLR = torch.nn.CrossEntropyLoss().to(device)
  noise = torch.randn(originalBATCH, NLAT, 1, 1, device=device)
  wali = torch.nn.DataParallel(wali, device_ids=list(range(GPUS))).to(device)
  # Debugging purposes :down
  # test_size(train_loader)
  EG_losses, C_losses, R_losses, Constrastive_losses = [], [], [], []
  curr_iter = C_iter = EG_iter = 0
  C_update, EG_update = True, False
  print('Training starts...')
  # torch.save(wali.module.state_dict(), f'{modeldir}/{MODEL}-init.ckpt')
  for curr_iter in range(ITER):
    for batch_idx, (x, _) in enumerate(train_loader, 1):
      # save model
      if batch_idx % MODELSAVE_ITER == 0:
        torch.save(wali.module.state_dict(), f'{modeldir}/{MODEL}-epoch-{curr_iter}-{batch_idx}.ckpt')
        print(f'Model saved to {modeldir}/{MODEL}-epoch-{curr_iter}-{batch_idx}.ckpt')
        logging.info(f"Model saved to {modeldir}/{MODEL}-epoch-{curr_iter}-{batch_idx}.ckpt")

        # # plot training loss curve
        # print(EG_losses, C_losses)
        # plt.figure(figsize=(10, 5))
        # plt.title('Training loss curve')
        # plt.plot(torch.tensor(EG_losses).cpu(), label='Encoder + Generator')
        # plt.plot(torch.tensor(C_losses).cpu(), label='Critic')
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig(f'{traindir}/loss_curve-{curr_iter}-{batch_idx}.png')
        # print("loss curve saved")
        # plot reconstructed images and samples
      if batch_idx % SAMPLESAVE_ITER == 0:
        print("Sample save to ", f'{traindir}/rect-{curr_iter}-{batch_idx}.png')
        wali.eval()
        real_x, rect_x = init_x[:originalBATCH], wali.module.reconstruct(init_x[:originalBATCH]).detach_()
        rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1) 
        rect_imgs = rect_imgs.view(originalBATCH * 2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
        genr_imgs = wali.module.generate([noise]).detach_().cpu()
        utils.save_image(rect_imgs * 0.5 + 0.5, f'{traindir}/rect{curr_iter}-{batch_idx}.png')
        utils.save_image(genr_imgs * 0.5 + 0.5, f'{traindir}/genr{curr_iter}-{batch_idx}.png')
        wali.train()
        print("rect, gen images saved")
      running_losses = [0, 0]
      # print("batch_idx: ", batch_idx)

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
        init_x = original_imgs
        print("init_x shape: ", init_x.shape)
        curr_iter += 1
        utils.save_image(init_x * 0.5 + 0.5, f'{traindir}/init-batch-imageSanityCheck{MODEL}.png')
      # Forward pass, get loss
      # Sample h from a prior distribution ~ N(0, 1)
      # original_imgs.size(0) = batch size
      # x[2] is the original image TODO
      h = torch.randn(x[2].size(0), H_DIM, 1, 1).to(device)
      # print("h shape in loop: ", h.shape)
      C_loss, EG_loss = wali(x, h, lamb=LAMBDAS, device=device, baseline = BASELINE)
      running_losses[0] += C_loss.sum()
      running_losses[1] += EG_loss.sum()
      # print("loss calculated C_loss: ", C_loss, "EG_loss: ",  EG_loss)
      if batch_idx % WRITER_ITER == 0:
        print('Epoch: {}, Batch: {} C_loss: {:.4f}, EG_loss: {:.4f}'.format(
          curr_iter, batch_idx, C_loss.sum(), EG_loss.sum()))
        # [Deprecated writer]
        # writer.add_scalar('C_loss', running_losses[0], (curr_iter - 1) * n_total_runs + batch_idx)
        # writer.add_scalar('EG_loss', running_losses[1], (curr_iter - 1) * n_total_runs + batch_idx)
        logging.info('C_loss: ' + str(running_losses[0]) + 'EG_loss: '+ str(running_losses[1]) + " epoch: " + str(curr_iter) + " batch"+ str(batch_idx))
      # C_update: C_loss and Reconstruction loss
      if C_update:
        print("C_update")
        optimizerC.zero_grad()
        C_loss.sum().backward()
        # C_losses.append(C_loss.sum())
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
        EG_loss.sum().backward()
        # EG_losses.append(EG_loss.sum())
        optimizerEG.step()
        EG_iter += 1
        if EG_iter == EG_ITERS:
          EG_iter = 0
          C_update, EG_update = True, False
          # curr_iter += 1 # not epoch, but iteration
        else:
          continue
      
        
      # # print training statistics
      # if curr_iter % 100 == 0:
      #   print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
      #     % (curr_iter, ITER, EG_loss.item(), C_loss.item()))

      



    
    
    # Outside of batch for loop ( simclr schedule updates)
    # if curr_iter >= 10:
    #     schedulerSimCLR.step()
  print("End of training")

def create_generator(H_DIM, IMAGE_SIZE):
  return Generator(IMAGE_SIZE, H_DIM, 8)

def create_critic(H_DIM, LEAK, DIM_D, IMAGE_SIZE):
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


def create_WALI(H_DIM, Z_DIM, LEAK, DIM_D, IMAGE_SIZE):
  E = ResNetSimCLR(H_DIM, Z_DIM)
  G = create_generator(H_DIM, IMAGE_SIZE)
  C = create_critic(H_DIM, LEAK, DIM_D, IMAGE_SIZE)
  wali = WALI(E, G, C)
  return wali


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