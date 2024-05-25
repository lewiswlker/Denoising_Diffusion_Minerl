# Denoising_Diffusion_Minerl
This research is based on the [MineRL 0.4.4](https://minerl.readthedocs.io/en/v0.4.4/) environment, which is based on the popular game Minecraft that provides a realistic, open-world environment containing multiple tasks. In this environment, intelligent agents are required to perform operations such as exploring, resource gathering, tool making and accomplishing diverse tasks. We apply the diffusion model to the MineRL environment, utilizing the powerful generative capabilities of the diffusion model to generate more realistic and varied game scene images, and the conditional diffusion model to generate action sequences for specific game tasks.

We referenced [the Pytorch version](https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file) of the diffusion model and the conditional diffusion model. And this is [the official TensorFlow version](https://github.com/hojonathanho/diffusion). 

<img src="https://github.com/lewiswlker/Denoising_Diffusion_Minerl/blob/main/images/game_scene.png" width="300">

## Installation
1.Clone this repo in the directory.
```
https://github.com/lewiswlker/Denoising_Diffusion_Minerl.git
```
2.Install the requirments.
```
cd Denoising_Diffusion_Minerl-main
pip install -r requirements.txt
```

## Datasets
We have two datasets, the Navigate dataset for training diffusion models to generate game scenes and the MineRLNavigate-v0 dataset for training conditional diffusion models to generate action vectors. You can download them through following methods:

1.BaiduDrive:
<br>-[Navigate](https://pan.baidu.com/s/18vsSW7eBcP_8ngMQar6fWA?pwd=w53e)
<br>-[MineRLNavigate-v0](https://pan.baidu.com/s/1KNyvmvtk8YrVPsumA6sRXg?pwd=f2l1)<br><br>

2.If you encounter difficulties with the above methods, you can download them from [Minerl's official website](https://minerl.readthedocs.io/en/v0.4.4/tutorials/data_sampling.html).
<br>   
Note that the dataset used to train the diffusion model is composed of images of 64*64 size, so you might need to extract them from the dataset downloaded from the official website, similar to the official code below:
```python
from minerl.data import BufferedBatchIter
data = minerl.data.make('MineRLObtainDiamond-v0')
iterator = BufferedBatchIter(data)
for current_state, action, reward, next_state, done \
    in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):

        # Print the POV @ the first step of the sequence
        print(current_state['pov'][0])

        # Print the final reward pf the sequence!
        print(reward[-1])

        # Check if final (next_state) is terminal.
        print(done[-1])

        # ... do something with the data.
        print("At the end of trajectories the length"
              "can be < max_sequence_len", len(reward))
```
However, for the conditional diffusion model, you don't need to extract the combination of images and actions anymore, because this part we have already implemented in the code.

## Training
### Train the diffusion model
To train the diffusion model, you can use `. /DDPM/train.py`, although you may need to modify the path to the dataset and the path to save the results for checkpoint&model. In addition, you can choose the complexity of the model and auxiliary modules according to your arithmetic power. However, there are some cases where they do not necessarily work better.
```python
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
```
Once the modifications are complete, run `python train.py` to train the diffusion model based on your training set.
---
### Train the conditional diffusion model
To train a conditional diffusion model, you need to directly use `. /CDDPM/CDDPM.py`. This is a python script that integrates modeling and training, so there is no need to write additional training scripts. You just need to change the directory of the dataset and the directory where the training results are saved in `def main()`.
```python
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
```
However, if the dataset you downloaded from the official website is not MineRLNavigate-v0, then you need to change it to your dataset name in `class MineRLDataset()`.


## Test
### Diffusion model

For the first task, you can directly use `. /DDPM/minerl_create.py` to generate the game scenario by calling your training results directly. Similarly, in the following code, you need to first change the path of the model you are calling to the folder where it is located, and the path where the generated results are saved.
```python
...
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
...
save_path = Path('/root/autodl-tmp/results_navigate/generate')
...
```
In fact, at the time of training, we have used `. /DDPM/fid.py` to compute the FID scores for each training phase and the model has been tested.
---
### Conditional diffusion model

For the second task, since in the paper we are forming conclusions by comparing its training effects with BASELINE. Therefore you can directly run `. /CDDPM/baseline.py` to compare the two. The paths are modified as in the previous training of the conditional diffusion model.

In addition, we also used t-SNE and PCA to downscale the image features extracted by the two models, so that we can see more intuitively how well the two models are able to extract image features. This can be done in part by running `. /CDDPM/t-sne.py`. You can also use Jupyter notebook, we provide `. /Conditional_DDPM/t-sne.ipynb`
