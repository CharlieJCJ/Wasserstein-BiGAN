import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from resnet import ResNet50
import numpy as np

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
  def __init__(self, mapping, shift=None):
    """ A deterministic conditional mapping. Used as an encoder or a generator.

    Args:
      mapping: An nn.Sequential module that maps the input to the output deterministically.
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
    output = self.mapping(input)
    if self.shift is not None:
      output = output + self.shift
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

  def encode(self, x):
    return self.E(x)

  def generate(self, z):
    return self.G(z)

  def reconstruct(self, x):
    return self.generate(self.encode(x))

  def criticize(self, x, z_hat, x_tilde, z):
    input_x = torch.cat((x, x_tilde), dim=0)
    input_z = torch.cat((z_hat, z), dim=0)
    output = self.C(input_x, input_z)
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

  # FIXME: check the variable names. 
  def forward(self, x, z, lamb=10):
    (h_hat, z_hat),  x_tilde = self.encode(x), self.generate(z) # FIXME not self.encode, it has two outputs.
    data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
    EG_loss = torch.mean(data_preds - sample_preds)
    C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
    Reconstruction_loss = nn.MSELoss(x, self.generate(z_hat))    # Need to check this - z is basically vector h? H_DIM, Z_DIM
    return C_loss, EG_loss, Reconstruction_loss

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

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        return h, z

############################################################################################################
# Continue merging simclr codebase here (contrastive learning dataset)
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
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
                                                          download=True)}

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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # print("My length is", len([self.base_transform(x) for i in range(self.n_views)] + [transform(x)]))
        # return [self.base_transform(x) for i in range(self.n_views)]
        return self.base_transform(x), self.base_transform(x), transform(x) # dataloader handles the rest


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
