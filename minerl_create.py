import torch
from torch import nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path
import torchvision.utils as vutils

# Initialize the model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)

diffusion = GaussianDiffusion(
    model=model,
    image_size=64,
    timesteps=1000,  # Number of timesteps
    sampling_timesteps=250,  # Number of timesteps for sampling
)

trainer = Trainer(
    diffusion_model=diffusion,
    folder="",
    train_batch_size=16,
    train_num_steps=100000,
    ema_decay=0.995,
    results_folder='/root/autodl-tmp/results_navigate_2',
    do_load=True,
    load_milestone='best'
)

# Use the EMA model to generate images
trainer.ema.ema_model.eval()
with torch.no_grad():
    # Generate images, set batch_size as needed
    generated_images = trainer.ema.ema_model.sample(batch_size=16)

# Save the generated images
save_path = Path('/root/autodl-tmp/results_navigate/generate')
save_path.mkdir(parents=True, exist_ok=True)
generated_images_path = save_path / 'generated_images.png'
vutils.save_image(generated_images, generated_images_path, nrow=4)
print(f"Generated images saved to {generated_images_path}")
