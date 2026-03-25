"""

The released experiment workflow is:
1. Offline pretraining on source-domain stress fields.
2. Online fine-tuning on target-domain stress fields.
3. Conditional sampling, visualization, and metric reporting.

Default data mapping for experiment group `1`:
- source domain: `Data/Data_1_trans.pt`
- target domain: `Data/Data_1_t_trans.pt`
- condition embeddings: `Data/classes_emb_trans_1.64.pt`
"""

import argparse
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import os
from torch.optim import Adam
from ema_pytorch import EMA
from torchvision import transforms as T, utils
from Scheduler import GradualWarmupScheduler
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from stress_data_utils import load_real_stress_data, make_synthetic_stress_data
from version import __version__
from tqdm.auto import tqdm

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# helpers functions

class Classifier(nn.Module):
    def __init__(self, height, width, num_classes, t_dim=1) -> None:
        super().__init__()
        self.linear_t = nn.Linear(t_dim, num_classes)
        self.linear_img = nn.Linear(height * width * 1, num_classes)

    def forward(self, x, t):
        """
        Args:
            x (_type_): [B, 3, N, N]
            t (_type_): [B,]

        Returns:
                logits [B, num_classes]
        """
        B = x.shape[0]
        t = t.view(B, 1)
        logits = self.linear_t(t.float()) + self.linear_img(x.view(x.shape[0], -1))
        return logits


def Fine_cond_fn(x, t, source, target, classifier_scale=1):
    assert target is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        mmd_loss = mmd_rbf(source, target)
        grad = torch.autograd.grad(mmd_loss, x_in)[0] * classifier_scale
        return grad


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class mmd_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target):
        loss = mmd_rbf(input, target)
        return loss


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class Dataset(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()


# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model

class Unet(nn.Module):
    def __init__(
            self,
            dim,
            classes_emb,
            cond_drop_prob=0.5,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=4,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            attn_dim_head=32,
            attn_heads=4
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings

        # self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(
            self,
            x,
            time,
            classes_emb,
            cond_drop_prob=None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance

        # classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            # image_size,
            height,
            width,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            ddim_sampling_eta=1.,
            offset_noise_strength=0.,
            min_snr_loss_weight=False,
            min_snr_gamma=5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        # self.image_size = image_size
        self.height = height
        self.width = width

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, cond_scale=6., rescaled_phi=0.7, clip_x_start=False):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale=cond_scale,
                                                          rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised=True):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def condition_mean(self, mmd_grad, mean, variance, x, t):
        new_mean = (
                mean.float() + variance * mmd_grad.float()
        )
        print("gradient: ", (variance * mmd_grad.float()).mean())
        return new_mean

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale=6., rescaled_phi=0.7, clip_denoised=True, mmd_grad=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, classes=classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised)
        # if exists(mmd_grad):
        # model_mean = self.condition_mean(mmd_grad, model_mean, variance, x, batched_times)

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale=6., rescaled_phi=0.7):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale=6., rescaled_phi=0.7, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale=cond_scale,
                                                             rescaled_phi=rescaled_phi, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale=6., rescaled_phi=0.7):
        import time as _time
        batch_size, height, width, channels = classes.shape[0], self.height, self.width, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        _t0 = _time.time()
        result = sample_fn(classes, (batch_size, channels, height, width), cond_scale, rescaled_phi)
        _sample_elapsed = _time.time() - _t0
        print(f'sample complete: total {_sample_elapsed:.2f}s, avg per sample {_sample_elapsed / batch_size:.2f}s')
        return result

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, _ = self.p_sample(img, i, classes)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # if self.offset_noise_strength > 0.:
        # offset_noise = torch.randn(x_start.shape[:2], device = self.device)
        # noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.smooth_l1_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, height, width = *img.shape, img.device, self.height, self.width
        assert h == height and w == width, f'height and width of image must be {height, width}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            augment_horizontal_flip=True,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=5000,
            num_samples=1,
            results_folder='./results',
            amp=False,
            classes=torch.empty(1, 128).cuda(0),
            mixed_precision_type='fp16',
            split_batches=False,
            convert_image_to=None,
            max_grad_norm=1.,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.classes = classes
        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        # self.image_size = diffusion_model.image_size
        self.height = diffusion_model.height
        self.width = diffusion_model.width

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        # assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=False)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        optimizer = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=train_num_steps,
                                                                     eta_min=0, last_epoch=-1)

        warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=2., warm_epoch=train_num_steps // 10,
                                                 after_scheduler=cosineScheduler)

        self.opt = optimizer
        self.warm = warmUpScheduler
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, type):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-all step-trans-{type}.pt'))

    def load(self, type):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-all step-trans-{type}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def fine(self, type):
        import time as _time
        accelerator = self.accelerator
        device = accelerator.device
        losses = []
        self.step = 0
        mmd = mmd_loss()
        _fine_start = _time.time()

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                data = next(self.dl).to(device)
                if self.step == 0 or self.step % 50 == 0:
                    # if self.step % 50 == 0:
                    G = torch.zeros(24, 3).to(device)
                    D = torch.zeros(24, 3).to(device)
                    w = data.shape[-1]
                    S = self.model.sample(classes=self.classes, cond_scale=8.)
                    for i in range(24):
                        D[i, 0] = data[i, 0, 0, 0]
                        D[i, 1] = data[i, 0, 0, w // 2]
                        D[i, 2] = data[i, 0, 0, -1]

                        G[i, 0] = S[i, 0, 0, 0]
                        G[i, 1] = S[i, 0, 0, w // 2]
                        G[i, 2] = S[i, 0, 0, -1]
                    G = G.requires_grad_(True)
                    D = D.requires_grad_(True)

                for _ in range(self.gradient_accumulate_every):
                    with self.accelerator.autocast():
                        loss_1 = self.model(data, classes=self.classes)
                        # print(loss_1)
                        loss_2 = mmd(G, D)
                        # print(loss_2)
                        loss = 0.01 * loss_1 + 0.99 * loss_2
                        # loss = loss_2
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        losses.append(loss.item())

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.warm.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        self.save(type)

                pbar.update(1)

        _fine_elapsed = _time.time() - _fine_start
        accelerator.print(f'fining complete, elapsed: {_fine_elapsed:.2f}s')
        plt.plot(losses, label='Fine Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Fine  Loss')
        plt.legend()
        # plt.savefig('./12.8_result.png', dpi=600)
        plt.show()

    def train(self, type):
        import time as _time
        accelerator = self.accelerator
        device = accelerator.device
        losses = []
        _train_start = _time.time()

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data, classes=self.classes)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        losses.append(loss.item())

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.warm.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        self.save(type)

                pbar.update(1)

        _train_elapsed = _time.time() - _train_start
        accelerator.print(f'training complete, elapsed: {_train_elapsed:.2f}s')
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training  Loss')
        plt.legend()
        # plt.savefig('./12.8_result.png', dpi=600)
        plt.show()


@torch.no_grad()
def classes_emb_conv(D_g):
    classes_data = torch.empty(D_g.shape[0], D_g.shape[1], 3)
    for i in range(D_g.shape[0]):
        classes_data[i, :, 0] = D_g[i, :, 0, 0]
        classes_data[i, :, 1] = D_g[i, :, 0, D_g.shape[-1] // 2]
        classes_data[i, :, 2] = D_g[i, :, 0, -1]

    # classes_data = rearrange(classes_data, 'b c n -> b 1 n c')
    # conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3).cuda(0)
    # classes_data_c = conv(classes_data)
    # classes_data_c = reduce(classes_data_c, 'b c h w -> b 1 w', reduction='mean')
    classes_data_c = F.interpolate(classes_data, size=64, mode='linear', align_corners=False)
    classes_emb = reduce(classes_data_c, 'b h w -> b w', reduction='mean')
    return classes_emb


def find_max_index(data):
    data = data
    width = data.shape[-1]
    index = torch.argmax(data)
    if (index + 1) % width == 0:
        i = (index + 1) // width - 1
        j = width - 1
    else:
        i = (index + 1) // width
        j = ((index + 1) % width) - 1
    return i, j


def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def Denominator(target):
    target_mean = torch.mean(target)
    tot = torch.sum((target - target_mean) ** 2)
    de = torch.sqrt(torch.mean(tot))
    return de


def RMAE(output, target):
    de = Denominator(target)
    a = torch.abs(torch.max(target - output))
    return a / de


def RRMSE(output, target):
    de = Denominator(target)
    MSE = nn.MSELoss()
    RMSE = torch.sqrt(MSE(target, output))
    return RMSE


def script_root() -> Path:
    return Path(__file__).resolve().parent


def default_paths(group_id: str):
    data_root = script_root() / 'Data'
    return {
        'source': data_root / f'Data_{group_id}_trans.pt',
        'target': data_root / f'Data_{group_id}_t_trans.pt',
        'classes': data_root / f'classes_emb_trans_{group_id}.64.pt'
    }


def build_argparser():
    parser = argparse.ArgumentParser(description='Stress-field transfer experiment from the paper.')
    parser.add_argument('--group-id', default='1', help='Experiment group id. Default: 1.')
    parser.add_argument('--source-data', type=str, default=None, help='Explicit path to source-domain data tensor.')
    parser.add_argument('--target-data', type=str, default=None, help='Explicit path to target-domain data tensor.')
    parser.add_argument('--classes-emb', type=str, default=None, help='Explicit path to condition embedding tensor.')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory for checkpoints and outputs.')
    parser.add_argument('--offline-steps', type=int, default=1000, help='Offline pretraining steps.')
    parser.add_argument('--online-steps', type=int, default=300, help='Online fine-tuning steps.')
    parser.add_argument('--sampling-steps', type=int, default=100, help='Sampling steps for diffusion.')
    parser.add_argument('--batch-size', type=int, default=24, help='Training batch size.')
    parser.add_argument('--cond-scale', type=float, default=8.0, help='Classifier-free guidance scale.')
    parser.add_argument('--offline-ckpt', type=str, default=None, help='Offline checkpoint tag. Default: group<id>_offline')
    parser.add_argument('--online-ckpt', type=str, default=None, help='Online checkpoint tag. Default: group<id>_online')
    parser.add_argument('--device', type=str, default=None, help='Torch device, e.g. cuda:0 or cpu.')
    parser.add_argument('--skip-plots', action='store_true', help='Disable matplotlib plots.')
    parser.add_argument('--sample-output', type=str, default=None, help='Path to save the predicted target-domain tensor.')
    parser.add_argument('--use-synthetic-data', action='store_true',
                        help='Use synthetic random stress data for public timing analysis.')
    parser.add_argument('--synthetic-seed', type=int, default=20260320,
                        help='Random seed for synthetic stress data.')
    parser.add_argument('--synthetic-num-samples', type=int, default=24,
                        help='Synthetic sample count.')
    parser.add_argument('--synthetic-height', type=int, default=24,
                        help='Synthetic stress field height.')
    parser.add_argument('--synthetic-width', type=int, default=88,
                        help='Synthetic stress field width.')
    parser.add_argument('--synthetic-cond-dim', type=int, default=64,
                        help='Synthetic condition embedding dimension.')
    return parser


def resolve_experiment_paths(args):
    if args.use_synthetic_data:
        return None, None, None, Path(args.results_dir) if args.results_dir else (script_root() / 'results')

    defaults = default_paths(args.group_id)
    source_path = Path(args.source_data) if args.source_data else defaults['source']
    target_path = Path(args.target_data) if args.target_data else defaults['target']
    classes_path = Path(args.classes_emb) if args.classes_emb else defaults['classes']
    results_dir = Path(args.results_dir) if args.results_dir else (script_root() / 'results')

    for path in (source_path, target_path, classes_path):
        if not path.exists():
            raise FileNotFoundError(f'Missing required input file: {path}')

    return source_path, target_path, classes_path, results_dir


def get_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_and_normalize_tensor(path: Path, device):
    data = torch.load(str(path), map_location=device).float().unsqueeze(1)
    max_data = torch.max(data)
    min_data = torch.min(data)
    data_norm = (data - min_data) / (max_data - min_data)
    return data, data_norm, min_data, max_data


def load_classes_embedding(path: Path, device):
    return torch.load(str(path), map_location=device).float()


def build_diffusion_model(classes_emb, channels, height, width, sampling_steps, device):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        classes_emb=classes_emb,
        cond_drop_prob=0.5,
        channels=channels,
        resnet_block_groups=4,
        attn_dim_head=32,
        attn_heads=4
    )

    diffusion = GaussianDiffusion(
        model,
        height=height,
        width=width,
        timesteps=1000,
        sampling_timesteps=sampling_steps,
        objective='pred_v'
    ).to(device)

    return model, diffusion


def build_trainer(diffusion, dataset, classes_emb, results_dir, train_lr, train_num_steps, batch_size):
    return Trainer(
        diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=train_lr,
        train_num_steps=train_num_steps,
        save_and_sample_every=train_num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        classes=classes_emb,
        results_folder=str(results_dir)
    )


def freeze_for_online_finetuning(model):
    for name, param in model.named_parameters():
        if 'ups' not in name:
            param.requires_grad = False


def ensure_transfer_checkpoint(
        diffusion,
        model,
        source_dataset,
        target_dataset,
        classes_emb,
        results_dir,
        offline_steps,
        online_steps,
        batch_size,
        offline_ckpt,
        online_ckpt
):
    online_trainer = build_trainer(
        diffusion=diffusion,
        dataset=target_dataset,
        classes_emb=classes_emb,
        results_dir=results_dir,
        train_lr=5e-5,
        train_num_steps=online_steps,
        batch_size=batch_size
    )

    try:
        online_trainer.load(online_ckpt)
        print(f'Loaded online checkpoint: {results_dir / f"model-all step-trans-{online_ckpt}.pt"}')
        return online_trainer
    except Exception as exc:
        print(f'Online checkpoint not available ({online_ckpt}): {exc}')

    offline_trainer = build_trainer(
        diffusion=diffusion,
        dataset=source_dataset,
        classes_emb=classes_emb,
        results_dir=results_dir,
        train_lr=1e-4,
        train_num_steps=offline_steps,
        batch_size=batch_size
    )

    try:
        offline_trainer.load(offline_ckpt)
        print(f'Loaded offline checkpoint: {results_dir / f"model-all step-trans-{offline_ckpt}.pt"}')
    except Exception as exc:
        print(f'Offline checkpoint not available ({offline_ckpt}): {exc}')
        print('Starting offline pretraining on source-domain data.')
        offline_trainer.train(type=offline_ckpt)

    online_trainer.load(offline_ckpt)
    freeze_for_online_finetuning(model)
    print('Starting online fine-tuning on target-domain data.')
    online_trainer.fine(type=online_ckpt)
    return online_trainer


def save_prediction_tensor(prediction, sample_output: Path):
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction.cpu(), str(sample_output))
    print(f'Saved predicted target-domain tensor to: {sample_output}')


def summarize_metrics(prediction, target):
    rmae_result = RMAE(prediction, target)
    rrmse_result = RRMSE(prediction, target)
    r2 = r2_score(prediction[:, 0, :, :], target[:, 0, :, :])
    print(f'Overall metrics -> R2: {r2.item():.4f}, RRMSE: {rrmse_result.item():.4f}, RMAE: {rmae_result.item():.4f}')

    for i in range(target.shape[0]):
        j, k = find_max_index(target[i, -1, :, :])
        pred_max = prediction[i:i + 1, -1, j, k]
        rmae_result = RMAE(prediction[i:i + 1, 0, :, :], target[i:i + 1, 0, :, :])
        rrmse_result = RRMSE(prediction[i:i + 1, 0, :, :], target[i:i + 1, 0, :, :])
        r2 = r2_score(prediction[i:i + 1, 0, :, :], target[i:i + 1, 0, :, :])
        re_max = torch.abs((pred_max - torch.max(target[i, -1, :, :])) / torch.max(target[i, -1, :, :]))
        print(
            f'Sample {i:02d} -> R2: {r2.item():.4f}, '
            f'RRMSE: {rrmse_result.item():.4f}, '
            f'RMAE: {rmae_result.item():.4f}, '
            f'RE_max: {re_max.item():.4f}'
        )


def plot_results(source_data, target_data, prediction):
    plt.figure(figsize=(12, 8))
    plt.xlabel("Axial distribution of rods", size=14)
    plt.ylabel("Radial distribution of rods", size=14)
    plt.imshow(target_data[-1, 0, :, :].cpu(), cmap='viridis', interpolation='bicubic')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.xlabel("Axial distribution of rods", size=14)
    plt.ylabel("Radial distribution of rods", size=14)
    plt.imshow(prediction[-1, 0, :, :].cpu(), cmap='viridis', interpolation='bicubic')
    plt.colorbar()
    plt.show()

    x = np.arange(0, 88, 1)
    y = np.arange(0, 24, 1)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(20, 8))

    sub_target = fig.add_subplot(132, projection='3d')
    for idx in (3, 14, -1):
        sub_target.plot_surface(X, Y, target_data[idx, 0, :, :].cpu(), rstride=1, cstride=1, cmap='rainbow')
    sub_target.set_xlabel(r"Axial distribution of rods")
    sub_target.set_ylabel(r"Radial distribution of rods")
    sub_target.set_zlabel(r"Stress distribution value")
    sub_target.set_title("Target stress distribution under three loading conditions")

    sub_source = fig.add_subplot(131, projection='3d')
    for idx in (3, 14, -1):
        sub_source.plot_surface(X, Y, source_data[idx, 0, :, :].cpu(), rstride=1, cstride=1, cmap='rainbow')
    sub_source.set_xlabel(r"Axial distribution of rods")
    sub_source.set_ylabel(r"Radial distribution of rods")
    sub_source.set_zlabel(r"Stress distribution value")
    sub_source.set_title("Original stress distribution under three loading conditions")

    sub_pred = fig.add_subplot(133, projection='3d')
    for idx in (3, 14, -1):
        sub_pred.plot_surface(X, Y, prediction[idx, 0, :, :].cpu(), rstride=1, cstride=1, cmap='rainbow')
    sub_pred.set_xlabel(r"Axial distribution of rods")
    sub_pred.set_ylabel(r"Radial distribution of rods")
    sub_pred.set_zlabel(r"Stress distribution value")
    sub_pred.set_title("Predicted stress distribution under three loading conditions")
    plt.show()

    for idx in (3, 16, 21):
        plt.plot(prediction[idx, 0, 22, :].cpu())
        plt.plot(target_data[idx, 0, 22, :].cpu())
        plt.show()


def main():
    parser = build_argparser()
    args = parser.parse_args()

    source_path, target_path, classes_path, results_dir = resolve_experiment_paths(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    offline_ckpt = args.offline_ckpt or f'group{args.group_id}_offline'
    online_ckpt = args.online_ckpt or f'group{args.group_id}_online'
    sample_output = Path(args.sample_output) if args.sample_output else (
        results_dir / f'predicted_group{args.group_id}_target.pt'
    )

    print('Experiment configuration:')
    print(f'  group id      : {args.group_id}')
    print(f'  source data   : {source_path}')
    print(f'  target data   : {target_path}')
    print(f'  classes emb   : {classes_path}')
    print(f'  results dir   : {results_dir}')
    print(f'  device        : {device}')
    print(f'  offline ckpt  : {offline_ckpt}')
    print(f'  online ckpt   : {online_ckpt}')

    if args.use_synthetic_data:
        source_data, target_data, classes_emb = make_synthetic_stress_data(
            num_samples=args.synthetic_num_samples,
            height=args.synthetic_height,
            width=args.synthetic_width,
            cond_dim=args.synthetic_cond_dim,
            seed=args.synthetic_seed,
            device=device
        )
        source_norm, _, _ = normalize_01(source_data)
        target_norm, min_target, max_target = normalize_01(target_data)
        print('  data mode     : synthetic benchmark')
    else:
        source_data, target_data, classes_emb, source_path, target_path, classes_path = load_real_stress_data(
            group_id=args.group_id,
            device=device,
            source_path=source_path,
            target_path=target_path,
            classes_path=classes_path
        )
        source_norm, _, _ = normalize_01(source_data)
        target_norm, min_target, max_target = normalize_01(target_data)
        print('  data mode     : real experiment data')

    print(f'Source-domain tensor shape : {tuple(source_norm.shape)}')
    print(f'Target-domain tensor shape : {tuple(target_norm.shape)}')
    print(f'Condition embedding shape  : {tuple(classes_emb.shape)}')

    source_dataset = Dataset(source_norm)
    target_dataset = Dataset(target_norm)

    model, diffusion = build_diffusion_model(
        classes_emb=classes_emb,
        channels=source_norm.shape[1],
        height=source_norm.shape[2],
        width=source_norm.shape[3],
        sampling_steps=args.sampling_steps,
        device=device
    )

    ensure_transfer_checkpoint(
        diffusion=diffusion,
        model=model,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        classes_emb=classes_emb,
        results_dir=results_dir,
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
        batch_size=args.batch_size,
        offline_ckpt=offline_ckpt,
        online_ckpt=online_ckpt
    )

    sampled_target = diffusion.sample(classes=classes_emb, cond_scale=args.cond_scale)
    predicted_target = sampled_target * (max_target - min_target) + min_target
    save_prediction_tensor(predicted_target, sample_output)
    summarize_metrics(predicted_target, target_data)

    if not args.skip_plots:
        plot_results(source_data, target_data, predicted_target)



if __name__ == '__main__':
    main()
