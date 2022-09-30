import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from torch.cuda.amp import GradScaler, autocast
from resnet import ResNet50
import numpy as np
from constants import *
def log_odds(p):
  p = torch.clamp(p.mean(dim=0), 1e-7, 1-1e-7)
  return torch.log(p / (1 - p))


class MaxOut(nn.Module):
  def __init__(self, k=2):
    """ MaxOut nonlinearity.
    
    Args:
      k: Number of linear pieces in the MaxOut opeartion. Default: 2
    """
    super().__init__()

    self.k = k

  def forward(self, input):
    output_dim = input.size(1) // self.k
    input = input.view(input.size(0), output_dim, self.k, input.size(2), input.size(3))
    output, _ = input.max(dim=2)
    return output

# A deterministic conditional mapping. Used as an encoder or a generator.
class DeterministicConditional(nn.Module):
  def __init__(self, mapping, shift=None, encoder = True):
    """ A deterministic conditional mapping. Used as an encoder or a generator.

    Args:
      mapping: An nn.Sequential module that maps the input to the output deterministically.
      shift: A pixel-wise shift added to the output of mapping. Default: None
    """
    super().__init__()
    self.encoder = encoder
    self.mapping = mapping
    self.shift = shift
    self.cv1 = ConvTranspose2d(H_DIM, DIM * 4, 4, 1, 0, bias=False)
    self.cv2 = ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False)
    self.cv3 = ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False)
    self.cv4 = ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False)
    self.bn1 = BatchNorm2d(DIM * 4)
    self.bn2 = BatchNorm2d(DIM * 2)
    self.bn3 = BatchNorm2d(DIM)
    self.rl = ReLU(inplace=True)
    self.tanh = Tanh()

  def set_shift(self, value):
    if self.shift is None:
      return
    assert list(self.shift.data.size()) == list(value.size())
    self.shift.data = value

  def forward(self, input):
    if self.encoder == True: 
      output = self.mapping(input)
    else: 
      output = self.cv1(input)
      output = self.bn1(output)
      output = self.rl(output)
      output = self.cv2(output)
      output = self.bn2(output)
      output = self.rl(output)
      output = self.cv3(output)
      output = self.bn3(output)
      output = self.rl(output)
      output = self.cv4(output)
      output = self.tanh(output)

    # nn.Sequential(
    # ConvTranspose2d(NLAT, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    # ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    # ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    # ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
    if self.shift is not None:
      output = output + self.shift
    # print(output.shape)
    return output


class GaussianConditional(nn.Module):
  def __init__(self, mapping, shift=None):
    """ A Gaussian conditional mapping. Used as an encoder or a generator.

    Args:
      mapping: An nn.Sequential module that maps the input to the parameters of the Gaussian.
      shift: A pixel-wise shift added to the output of mapping. Default: None
    """
    super().__init__()

    self.mapping = mapping
    self.shift = shift

  def set_shift(self, value):
    if self.shift is None:
      return
    assert list(self.shift.data.size()) == list(value.size())
    self.shift.data = value

  def forward(self, input):
    params = self.mapping(input)
    nlatent = params.size(1) // 2
    mu, log_sigma = params[:, :nlatent], params[:, nlatent:]
    sigma = log_sigma.exp()
    eps = torch.randn(mu.size()).to(input.device)
    output = mu + sigma * eps
    if self.shift is not None:
      output = output + self.shift
    return output


class JointCritic(nn.Module):
  def __init__(self, x_mapping, z_mapping, joint_mapping):
    """ A joint Wasserstein critic function.

    Args:
      x_mapping: An nn.Sequential module that processes x.
      z_mapping: An nn.Sequential module that processes z.
      joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
    """
    super().__init__()

    self.x_net = x_mapping
    self.z_net = z_mapping
    self.joint_net = joint_mapping

  def forward(self, x, z):
    assert x.size(0) == z.size(0)
    x_out = self.x_net(x) # Discriminator
    z_out = self.z_net(z) # z mapping
    # print("x", x.shape, "z", z.shape)
    # print("x_out", x_out.shape, "z_out", z_out.shape)
    joint_input = torch.cat((x_out, z_out), dim=1) # Concatenate
    output = self.joint_net(joint_input)
    return output

# End-to-end model here
class WALI(nn.Module):
  def __init__(self, E, G, C):
    """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.

    Args:
      E: Encoder p(z|x).
      G: Generator p(x|z).
      C: Wasserstein critic function f(x, z).
    """
    super().__init__()

    self.E = E
    self.G = G
    self.C = C

  def get_encoder_parameters(self):
    return self.E.parameters()

  def get_generator_parameters(self):
    return self.G.parameters()

  def get_critic_parameters(self):
    return self.C.parameters()

  # Returns a tuple, because we changed how encoder outputs
  def encode(self, x):
    return self.E(x)

  def generate(self, z):
    return self.G(z)

  def reconstruct(self, x):
    h, z = self.encode(x)
    output = self.generate([h.data])
    # print("h", h.shape, "z", z.shape, "output", output.shape)
    return output

  def criticize(self, x, z_hat, x_tilde, z):
    input_x = torch.cat((x, x_tilde), dim=0)
    input_z = torch.cat((z_hat, z), dim=0)
    output = self.C(input_x, input_z) # TODO: check output here
    data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
    return data_preds, sample_preds

  def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
    bsize = x.size(0)
    eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
    intp_x = eps * x + (1 - eps) * x_tilde
    intp_z = eps * z_hat + (1 - eps) * z
    intp_x.requires_grad = True
    intp_z.requires_grad = True
    C_intp_loss = self.C(intp_x, intp_z).sum()
    grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
    grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
    grads = torch.cat((grads_x, grads_z), dim=1)
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty

  # Check what x is in this context. It can add another variable that checks for whether it's for simclr or EG/C.
  # I mean I can do it both by passing into the entire x. 
  # FIXME: check the variable names. 
  # Now: I pass in original images
  # Edit: x is now 3 dimensional
  def forward(self, x, h, lamb=10, device="cuda", baseline = False):
    # x_tilde is the generated image
    transformed_imgs = torch.cat([x[0], x[1]], dim=0) # expecting 512 * 3 * 32 * 32 (batch size is 256)
    original_imgs = x[2]
    transformed_imgs = transformed_imgs.to(device)
    original_imgs = original_imgs.to(device)
    # print("x: ", original_imgs.shape, "h: ", h.shape)
    (h_hat, z_hat),  x_tilde = self.encode(original_imgs), self.generate([h]) # FIXME not self.encode, it has two outputs. 
                                                                # We don't need z_hat in this case.
    # print(h_hat.shape, z_hat.shape, x_tilde.shape)
    if baseline:
        print("Baseline used")
        data_preds, sample_preds = self.criticize(original_imgs, h_hat, x_tilde, h) 
        EG_loss = torch.mean(data_preds - sample_preds).double()
        C_loss = -EG_loss + lamb * self.calculate_grad_penalty(original_imgs.data, h_hat.data, x_tilde.data, h.data)
        return C_loss , EG_loss
    else:
      criterionSimCLR = torch.nn.CrossEntropyLoss().to(device)
      with autocast(enabled=True):
          # print("get constrastive loss")
          # use forward
          __, features = self.encode(transformed_imgs) # only use z
          logits, labels = info_nce_loss(features, device)
          Constrastive_loss = criterionSimCLR(logits, labels)
      data_preds, sample_preds = self.criticize(original_imgs, h_hat, x_tilde, h) 
      EG_loss = torch.mean(data_preds - sample_preds)
      C_loss = -EG_loss + lamb * self.calculate_grad_penalty(original_imgs.data, h_hat.data, x_tilde.data, h.data)
      Reconstruction_loss = nn.MSELoss()(original_imgs, self.generate([h_hat]))    # Need to check this - z is basically vector h? H_DIM, Z_DIM
      return C_loss + Reconstruction_loss, EG_loss + Constrastive_loss
def info_nce_loss(features, device):
    features = features.reshape((features.shape[0], features.shape[1]))
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
############################################################################################################
# Merging simclr codebase here
class Projection(nn.Module):
  """
  Creates projection head
  Args:
    n_in (int): Number of input features
    n_hidden (int): Number of hidden features
    n_out (int): Number of output features
    use_bn (bool): Whether to use batch norm
  """
  def __init__(self, n_in: int, n_hidden: int, n_out: int,
               use_bn: bool = True):
    super().__init__()
    
    # No point in using bias if we've batch norm
    self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
    self.bn = nn.BatchNorm1d(n_hidden) if use_bn else nn.Identity()
    self.relu = nn.ReLU()
    # No bias for the final linear layer
    self.lin2 = nn.Linear(n_hidden, n_out, bias=False)
  
  def forward(self, x):
    x = self.lin1(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.lin2(x)
    return x

class ResNetSimCLR(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(ResNetSimCLR, self).__init__()
        
        # ResNet backbone
        self.backbone = ResNet50()
        self.projection = Projection(h_dim, 512,
                                 z_dim, False)
    # We've changed the encoder(.) in WALI class
    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        # print("Projection z", z.shape)
        h = h.reshape((h.shape[0], h.shape[1], 1, 1))
        z = z.reshape((z.shape[0], z.shape[1], 1, 1))
        return h, z

############################################################################################################
# Continue merging simclr codebase here (contrastive learning dataset)
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, flag_resize = False):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        if flag_resize:
          data_transforms = transforms.Compose([transforms.Resize(size=size),
                                                transforms.RandomResizedCrop(size=size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                transforms.ToTensor()])
        else:
          data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True), 
                          'mnist': lambda: datasets.MNIST(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32, flag_resize = True),
                                                                  n_views),
                                                              download=True)                                
                                                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        x = x.convert('RGB')
        
        transform = transforms.Compose([
          transforms.Resize(size=32),
          transforms.ToTensor(),
          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        # print("My length is", len([self.base_transform(x) for i in range(self.n_views)] + [transform(x)]))
        # return [self.base_transform(x) for i in range(self.n_views)]
        a = self.base_transform(x)
        b, c = self.base_transform(x), transform(x)
        # print("a", a.shape, "b", b.shape, "c", c.shape)
        return a, b, c # dataloader handles the rest


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
