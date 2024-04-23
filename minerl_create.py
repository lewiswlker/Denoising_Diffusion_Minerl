import torch
from torch import nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path
import torchvision.utils as vutils

# 初始化模型
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,
)


trainer = Trainer(
    diffusion_model=diffusion,
    folder="",
    train_batch_size=16,
    train_num_steps=100000,
    ema_decay=0.995,
    results_folder='/root/autodl-tmp/results_navigate_2',
    do_load=True,
    load_milestone='best'  # 假设最优模型保存为'best'
)

# 使用EMA模型进行图像生成
trainer.ema.ema_model.eval()
with torch.no_grad():
    # 生成图像，这里的batch_size根据需要设置
    generated_images = trainer.ema.ema_model.sample(batch_size=16)

# 保存生成的图像
save_path = Path('/root/autodl-tmp/results_navigate_2/generate')
save_path.mkdir(parents=True, exist_ok=True)
generated_images_path = save_path / 'generated_images.png'
vutils.save_image(generated_images, generated_images_path, nrow=4)
print(f"Generated images saved to {generated_images_path}")
