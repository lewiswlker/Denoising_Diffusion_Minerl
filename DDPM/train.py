from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 96,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250
)

trainer = Trainer(
    diffusion,
    '/root/autodl-tmp/navigate_selected',
    train_batch_size = 64,
    train_lr = 8e-5,
    train_num_steps = 150000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,             # whether to calculate fid during training
    save_and_sample_every = 5000,
    results_folder = '/root/autodl-tmp/results',
    save_best_and_latest_only = True,
    num_fid_samples = 2000,
    do_load = False,
    load_milestone='latest'
)

trainer.train()
