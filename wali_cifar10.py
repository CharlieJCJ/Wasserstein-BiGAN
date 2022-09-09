# Two optimizer (EG + C) version, instead of three (plus simclr)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from util import JointCritic, WALI, ResNetSimCLR, ContrastiveLearningDataset
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from stylegan2 import Generator, Discriminator


cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


# training hyperparameters
N_VIEW = 2
BATCH_SIZE = 256
ITER = 200000 # Number of epochs to train for
IMAGE_SIZE = 32
NUM_CHANNELS = 3
H_DIM = 512
Z_DIM = 128
DIM_D = 2048 # Need to check the size in stylegan2.py using test()
NLAT = 512
LEAK = 0.2

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
  # kernal size is 1, stride is 1, padding is 0. 
  z_mapping = nn.Sequential(
    Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))
  
  # DIM_D = 2048, dimension of the output of the discriminator
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
  
  # FIXME - wali.get_encoder_parameters() might be the entire resnet + MLP. - FIXED
  optimizerEG = Adam(list(wali.E.backbone.parameters()) + list(wali.get_generator_parameters()), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
  optimizerC = Adam(wali.get_critic_parameters(), 
    lr=LEARNING_RATE, betas=(BETA1, BETA2))

  # Update: we don't need separate SimCLR Encoder and scheduler? 
  # SimCLR Encoder and training scheduler
  optimizerSimCLR = torch.optim.Adam(wali.get_encoder_parameters(), 0.0003, weight_decay=1e-4)

  schedulerSimCLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerSimCLR, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
  scalerSimCLR = GradScaler(enabled=True)
  criterionSimCLR = torch.nn.CrossEntropyLoss().to(device)
  # Legacy code
  # transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  # svhn = datasets.CIFAR10(CIFAR_PATH, download=True, train=True, transform=transform)
  # loader = data.DataLoader(svhn, BATCH_SIZE, shuffle=True, num_workers=2)
  noise = torch.randn(64, NLAT, 1, 1, device=device)
  
  # Debugging purposes :down
  # test_size(train_loader)
  EG_losses, C_losses, R_losses = [], [], []
  curr_iter = C_iter = EG_iter = 0
  C_update, EG_update = True, False
  print('Training starts...')
  
  while curr_iter < ITER:
    for batch_idx, (x, _) in enumerate(train_loader, 1):
      print("batch_idx: ", batch_idx)

      # Transformed, original image
      transformed_imgs = torch.cat([x[0], x[1]], dim=0) # expecting 512 * 3 * 32 * 32 (batch size is 256)
      original_imgs = x[2]
      transformed_imgs = transformed_imgs.to(device)
      original_imgs = original_imgs.to(device)

      with autocast(enabled=True):
        __, features = wali.encode(transformed_imgs) # only use z
        logits, labels = info_nce_loss(features, device)
        loss = criterionSimCLR(logits, labels)

      optimizerSimCLR.zero_grad()

      scalerSimCLR.scale(loss).backward()

      scalerSimCLR.step(optimizerSimCLR)
      scalerSimCLR.update()
      
      ############################
      # Starting BiGAN procedures
    
      # Curr_iter starts from one, and stores to init_x
      if curr_iter == 0:
        init_x = original_imgs
        curr_iter += 1

      # Sample z from a prior distribution ~ N(0, 1)
      z = torch.randn(original_imgs.size(0), NLAT, 1, 1).to(device)
      C_loss, EG_loss, R_loss= wali(x, z, lamb=LAMBDA)
      
      # print("batch_idx: ", C_loss, EG_loss)


      # Update critic: C loss and *Reconstruction loss*
      if C_update:
        optimizerC.zero_grad()
        C_loss.backward()
        C_losses.append(C_loss.item())
        # add reconstruction loss backward() here
        R_loss.backward()
        R_losses.append(R_loss.item())
        optimizerC.step()
        C_iter += 1
        if C_iter == C_ITERS:
          C_iter = 0
          C_update, EG_update = False, True
        continue
      
      # Update generator: EG loss (consist of G loss and E loss and *Constrastive loss*)
      if EG_update:
        optimizerEG.zero_grad()
        EG_loss.backward()
        EG_losses.append(EG_loss.item())
        optimizerEG.step()
        EG_iter += 1
        # construction loss backward() here
        if EG_iter == EG_ITERS:
          EG_iter = 0
          C_update, EG_update = True, False
          curr_iter += 1
        else:
          continue

      # print training statistics
      if curr_iter % 100 == 0:
        print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
          % (curr_iter, ITER, EG_loss.item(), C_loss.item()))

        # plot reconstructed images and samples
        wali.eval()
        real_x, rect_x = init_x[:32], wali.reconstruct(init_x[:32]).detach_()
        rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1) 
        rect_imgs = rect_imgs.view(64, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
        genr_imgs = wali.generate(noise).detach_().cpu()
        utils.save_image(rect_imgs * 0.5 + 0.5, 'cifar10/rect%d.png' % curr_iter)
        utils.save_image(genr_imgs * 0.5 + 0.5, 'cifar10/genr%d.png' % curr_iter)
        wali.train()

      # save model
      if curr_iter % (ITER // 10) == 0:
        torch.save(wali.state_dict(), 'cifar10/models/%d.ckpt' % curr_iter)
    # Outside of batch for loop (for simclr schedule updates)
    if curr_iter >= 10:
        schedulerSimCLR.step()
  # plot training loss curve
  plt.figure(figsize=(10, 5))
  plt.title('Training loss curve')
  plt.plot(EG_losses, label='Encoder + Generator')
  plt.plot(C_losses, label='Critic')
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('cifar10/loss_curve.png')

def test_size(train_loader):
  for batch_idx, (x, _) in enumerate(train_loader, 1):
    # x[0] is the first view of the first batch
    # x[1] is the second view of the first batch
    # x[2] is the original iamges of the first batch
    print("batch_idx: ", batch_idx)
    print("x: ", x[0], x[1], x[2])
    print("x: ", x[0].shape, x[1].shape, x[2].shape)
    break

# Calculate simclr loss - in simclr training pipeline
def info_nce_loss(features, device):

    labels = torch.cat([torch.arange(BATCH_SIZE) for i in range(N_VIEW)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / 0.07
    return logits, labels
if __name__ == "__main__":
  main()