import math
import numpy as np
from pathlib import Path
from random import randint

from collections import namedtuple

import torch
from torch import nn
from torch.optim import AdamW

import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm
from torch.utils.data import random_split
import minerl
import os
from minerl.data import BufferedBatchIter
import matplotlib.pyplot as plt

def encode_action(action):
    # 离散动作直接使用值
    discrete_actions = np.array([
        action['attack'], action['back'], action['forward'],
        action['jump'], action['left'], action['right'],
        action['sneak'], action['sprint']
    ]).flatten()  # 确保是一维数组

    # 处理连续动作：'camera'，直接使用数值
    # 由于camera的范围是[-180, 180]，我们可以将其标准化到[-1, 1]以提高稳定性
    camera_actions = np.array(action['camera']).flatten() / 180.0  # 确保是一维数组，并进行标准化

    # 处理枚举动作：'place'
    # 我们将'dirt'映射为[1]，'none'映射为[0]
    place_action = np.array([1]) if action['place'] == 'dirt' else np.array([0])

    # 将所有编码后的动作拼接为一个向量
    encoded_action = np.concatenate([discrete_actions, camera_actions, place_action])

    return encoded_action

class MineRLDataset(Dataset):
    def __init__(self, environment='MineRLNavigate-v0', transform=None, img_size=64, max_samples=20000):
        self.img_size = img_size
        self.data = minerl.data.make(environment)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.samples = []

        iterator = BufferedBatchIter(self.data)
        count = 0  # 初始化计数器
        for current_state, action, _, _, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
            if count >= max_samples:  # 检查是否达到了最大样本数
                break  # 如果是，则退出循环
            pov_image = current_state['pov'][0]  # 取序列的第一个图像
            encoded_action = encode_action(action)
            self.samples.append((pov_image, encoded_action))
            count += 1  # 更新计数器

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pov_image, encoded_action = self.samples[idx]
        pov_image = self.transform(pov_image)  # 应用预处理转换
        encoded_action = torch.tensor(encoded_action, dtype=torch.float32)
        return pov_image, encoded_action

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions
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

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        device = t.device
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ActionGeneratorCNN(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), pool=5, channels=3, action_dim=11, time_emb_dim=8):
        super().__init__()
        self.channels = channels
        self.action_dim = action_dim
        self.time_emb_dim = time_emb_dim
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Initialize dimensions
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.features = nn.Sequential()
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            self.features.add_module(f"conv{i}", nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1))
            self.features.add_module(f"batchnorm{i}", nn.BatchNorm2d(out_dim))
            self.features.add_module(f"relu{i}", nn.ReLU(inplace=True))

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))

        # Final fully connected layer
        self.final_fc = nn.Linear(out_dim * pool * pool + time_emb_dim + action_dim, action_dim)

    def forward(self, img, action, timesteps):
        img_features = self.features(img)
        img_features = self.global_pool(img_features)
        img_features = torch.flatten(img_features, 1)
        
        time_emb = self.time_emb(timesteps.to(device=img.device, dtype=torch.long))

        combined_features = torch.cat([img_features, time_emb, action.float()], dim=1)
        
        action_output = self.final_fc(combined_features)
        return action_output

    def extract_features(self, img):
        # Only the image features are processed and returned
        img_features = self.features(img)
        img_features = self.global_pool(img_features)
        img_features = torch.flatten(img_features, 1)
        return img_features

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


def process_actions(action_output):
    # 假设前8位和第11位是离散动作
    discrete_actions = torch.sigmoid(action_output[:, :8])
    # discrete_actions = action_output[:, :8]
    discrete_actions = (discrete_actions > 0.5)

    # 对第9位和第10位连续动作应用tanh激活函数
    continuous_actions = torch.tanh(action_output[:, 8:10])
    # continuous_actions = action_output[:, 8:10]
    # 合并处理后的动作向量
    processed_action_output = torch.cat(
        [discrete_actions, continuous_actions, discrete_actions[:, -1].unsqueeze(1)], dim=1)
    return processed_action_output


class ActionGaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size=64,  # 输入图像的尺寸
            shape=(1, 11),
            timesteps=1000,  # 扩散时间步数
            sampling_timesteps=250,  # 采样使用的时间步数，默认为None，即与timesteps相同
            objective='pred_x0',  # 扩散过程的目标类型
            beta_schedule='cosine',  # β值的调度策略
            ddim_sampling_eta=1.0,  # DDIM采样中的η值
            offset_noise_strength=0.0,  # 偏移噪声强度
            min_snr_loss_weight=False,  # 是否使用最小SNR损失权重
            min_snr_gamma=5  # 最小SNR损失权重的γ值
    ):
        super().__init__()
        self.shape = shape
        self.model = model
        self.image_size = image_size
        self.objective = objective

        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"未知的beta_schedule: {beta_schedule}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.offset_noise_strength = offset_noise_strength

        # 注册必要的缓冲区以便于在设备间移动和保持值不变
        self.safe_register_buffer('betas', self.betas)
        self.safe_register_buffer('alphas', self.alphas)
        self.safe_register_buffer('alphas_cumprod', self.alphas_cumprod)
        self.safe_register_buffer('alphas_cumprod_prev', self.alphas_cumprod_prev)

        # 在赋值之前，直接注册计算出的变量为缓冲区
        self.safe_register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.safe_register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.safe_register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - self.alphas_cumprod))
        self.safe_register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod))
        self.safe_register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod - 1))
        self.safe_register_buffer('posterior_variance',
                                  self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.safe_register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.safe_register_buffer('posterior_mean_coef1',
                                  self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.safe_register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod))

        # 根据目标类型初始化损失权重
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        maybe_clipped_snr = snr.clone().clamp_(max=min_snr_gamma) if min_snr_loss_weight else snr
        if objective == 'pred_noise':
            self.loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            self.loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            self.loss_weight = maybe_clipped_snr / (snr + 1)
        self.safe_register_buffer('loss_weight', self.loss_weight)

    def safe_register_buffer(self, name, tensor):
        # 如果模型已有该属性，先删除
        if hasattr(self, name):
            delattr(self, name)
        # 注册缓冲区
        self.register_buffer(name, tensor)
    
    def extract_image_features(self, img):
        return self.model.extract_features(img)
    
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
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
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

    def model_predictions(self, x, t, cond_image):
        model_output = self.model(cond_image, x, t)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise=pred_noise, pred_x_start=x_start)

    def p_mean_variance(self, x, t, cond_image):
        # 这里假设`x`是动作向量
        preds = self.model_predictions(x, t, cond_image)
        x_start = preds.pred_x_start

        # 计算后验均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_image):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, cond_image=cond_image,
                                                                          )
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_action = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_action, x_start

    @torch.no_grad()
    def p_sample_loop(self, cond_image, shape):
        batch, device = shape[0], self.betas.device

        action = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            action, x_start = self.p_sample(action, t, cond_image)

        action = process_actions(action)

        return action

    @torch.no_grad()
    def ddim_sample(self, cond_image):
        shape = self.shape
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        action = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(action, time_cond, cond_image)

            if time_next < 0:
                action = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(action)

            action = x_start * alpha_next.sqrt() + \
                     c * pred_noise + \
                     sigma * noise

        action = process_actions(action)
        return action

    @torch.no_grad()
    def sample(self, cond_image):
        shape = self.shape
        # `shape`参数现在应直接反映动作向量的形状
        sample_fn = self.ddim_sample if self.is_ddim_sampling else self.p_sample_loop
        # 生成动作向量时需要传入条件图像
        return sample_fn(cond_image)

    @torch.no_grad()
    def interpolate(self, cond_image, action1, action2, t=None, lam=0.5):
        device = action1.device
        t = default(t, self.num_timesteps - 1)

        assert action1.shape == action2.shape

        # 为每个动作生成时间步向量
        t_batched = torch.full((action1.size(0),), t, device=device, dtype=torch.long)

        # 生成两个动作向量的噪声版本
        noise_action1 = self.q_sample(action1, t_batched)
        noise_action2 = self.q_sample(action2, t_batched)

        # 插值生成新的动作向量
        interpolated_action = (1 - lam) * noise_action1 + lam * noise_action2

        # 逆过程生成清晰的动作向量，引入条件图像
        for i in reversed(range(0, t + 1)):
            interpolated_action, _ = self.p_sample(cond_image, interpolated_action, i)

        return interpolated_action

    @torch.no_grad()
    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, img, action_start, t, *, noise=None):
        b, action_dim = action_start.shape
        noise = default(noise, lambda: torch.randn_like(action_start))

        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        model_out = self.model(img, action_noisy, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = action_start
        elif self.objective == 'pred_v':
            v = self.predict_v(action_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 分别处理离散和连续动作
        discrete_indices = list(range(8)) + [10]  # 离散动作索引
        continuous_indices = [8, 9]  # 连续动作索引

        discrete_target = target[:, discrete_indices]
        continuous_target = target[:, continuous_indices]

        discrete_pred = model_out[:, discrete_indices]
        continuous_pred = model_out[:, continuous_indices]

        # 使用BCELoss计算离散动作损失
        bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
        discrete_loss = bce_with_logits_loss(discrete_pred, discrete_target).mean()

        # 使用MSELoss计算连续动作损失
        mse_loss = nn.MSELoss(reduction='none')
        continuous_loss = mse_loss(continuous_pred, continuous_target).mean()

        # 组合损失
        total_loss = discrete_loss * 0.1 + continuous_loss * 0.9
        # total_loss = discrete_loss
        # total_loss = continuous_loss
        # print('discrete_loss =',discrete_loss,'  continuous_loss =',continuous_loss)
        # 应用损失权重
        # loss_weight = extract(self.loss_weight, t, total_loss.shape)
        # weighted_loss = total_loss * loss_weight
        weighted_loss = total_loss * 1
        return weighted_loss.mean()

    def forward(self, img, action, t=None, *args, **kwargs):
        # 确认图像尺寸
        b, _, h, w = img.shape
        assert h == self.image_size and w == self.image_size, f'Image height and width must be {self.image_size}'

        # 如果没有提供时间步t，则随机选择一个
        t = t if t is not None else torch.randint(0, self.num_timesteps, (b,), device=img.device).long()

        return self.p_losses(img, action, t, *args, **kwargs)


class ConditionalDiffusionTrainer:
    def __init__(self, model, diffusion_model, dataset, test_dataset,
                 train_batch_size=32, val_batch_size=16, train_lr=1e-4,
                 weight_decay=0.01, num_epochs=100,
                 save_and_sample_every=10, test_every_n_epochs=1,
                 results_folder='/root/CDDPM/results', device=None, do_load=False):
        self.model = model
        self.diffusion_model = diffusion_model
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_lr = train_lr
        self.num_epochs = num_epochs
        self.save_and_sample_every = save_and_sample_every
        self.test_every_n_epochs = test_every_n_epochs
        self.results_folder = Path(results_folder)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_loss = float('inf')

        self.model.to(self.device)
        self.diffusion_model.to(self.device)

        self.train_dataloader = DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, pin_memory=True)

        self.opt = AdamW(self.diffusion_model.parameters(), lr=self.train_lr, weight_decay=weight_decay)

        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.log_file = self.results_folder / 'training_log.txt'
        with open(self.log_file, 'w') as f:
            f.write("Epoch,Phase,Loss\n")
        if do_load:
            self.load_model(str(self.results_folder / 'model_latest.pt'))

    def train_and_test(self):
        for epoch in range(0, self.num_epochs + 1):
            self.train_one_epoch(epoch)
            if epoch % self.test_every_n_epochs == 0:
                test_loss = self.test(epoch)
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    self.save_model('model_best.pth')
            if epoch % self.save_and_sample_every == 0:
                self.save_model('model_latest.pth')

    def train_one_epoch(self, epoch):
        self.diffusion_model.train()
        total_loss = 0
        for images, actions in self.train_dataloader:
            images, actions = images.to(self.device), actions.to(self.device)
            self.opt.zero_grad()
            loss = self.diffusion_model(images, actions)
            loss.backward()
            self.opt.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(self.train_dataloader.dataset)
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},Train,{avg_loss:.6f}\n")
        print(f'Epoch {epoch}: Train Loss = {avg_loss:.6f}')

    def test(self, epoch):
        self.diffusion_model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, actions in self.test_dataloader:
                images, actions = images.to(self.device), actions.to(self.device)
                loss = self.diffusion_model(images, actions)
                total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(self.test_dataloader.dataset)
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},Test,{avg_loss:.6f}\n")
        print(f'Epoch {epoch}: Test Loss = {avg_loss:.6f}')
        return avg_loss

    def save_model(self, filename):
        model_path = self.results_folder / filename
        torch.save({
            'model_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
        }, model_path)
        print(f"Saved model to {model_path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model loaded")

class TestDiffusionTrainer:
    def __init__(self, model, test_dataset, device=None):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False, pin_memory=True)
        self.model.to(self.device)

    def test(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, actions in self.test_dataloader:
                images, actions = images.to(self.device), actions.to(self.device)
                loss = self.model(images, actions)
                total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(self.test_dataset)
        print(f"Average Test Loss: {avg_loss:.4f}")
        return avg_loss

def main():
    os.environ['MINERL_DATA_ROOT'] = '/root/autodl-tmp'
    results_path = '/root/CDDPM/results'
    Path(results_path).mkdir(parents=True, exist_ok=True)

    full_dataset = MineRLDataset()
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActionGeneratorCNN(dim=64, dim_mults=(1, 2, 4, 8), channels=3, action_dim=11).to(device)
    diffusion_model = ActionGaussianDiffusion(model=model, image_size=64, shape=(1, 11), timesteps=1000,
                                              sampling_timesteps=250, objective='pred_x0').to(device)
    trainer = ConditionalDiffusionTrainer(model, diffusion_model, train_dataset, test_dataset,
                                          train_batch_size=64, val_batch_size=16, train_lr=1e-6,
                                          weight_decay=0.01, num_epochs=500, save_and_sample_every=10,
                                          test_every_n_epochs=1, results_folder=results_path, device=device)
    
    best_model_path = os.path.join(results_path, 'model_best.pth')
    if os.path.exists(best_model_path):
        print("Loading and testing the best model...")
        best_model = ActionGaussianDiffusion(model=model, image_size=64, shape=(1, 11), timesteps=1000,
                                              sampling_timesteps=250, objective='pred_x0').to(device)
        best_model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        tester = TestDiffusionTrainer(best_model, test_dataset, device)
        tester.test()
    else:
        print("No best model found. Training new model...")
        trainer.train_and_test()
        print("Training completed. Testing the best model...")
        # Attempt to test after training if model_best.pth was created
        if os.path.exists(best_model_path):
            best_model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
            tester = TestDiffusionTrainer(best_model, test_dataset, device)
            tester.test()

if __name__ == '__main__':
    main()