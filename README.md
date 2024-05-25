# Denoising_Diffusion_Minerl
This research is based on the [MineRL 0.4.4](https://minerl.readthedocs.io/en/v0.4.4/) environment, which is based on the popular game Minecraft that provides a realistic, open-world environment containing multiple tasks. In this environment, intelligent agents are required to perform operations such as exploring, resource gathering, tool making and accomplishing diverse tasks. We apply the diffusion model to the MineRL environment, utilizing the powerful generative capabilities of the diffusion model to generate more realistic and varied game scene images, and the conditional diffusion model to generate action sequences for specific game tasks.

We referenced [the Pytorch version of the diffusion model and the conditional diffusion model](https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file). But this is [the official TensorFlow version](https://github.com/hojonathanho/diffusion). 

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
We have two datasets, the Navigate dataset for training diffusion models to generate game scenes and the MineRLNavigate-v0 dataset for training conditional diffusion models to generate action vectors. You can download them via:
1. Download via BaiduDrive:
  -[Navigate](https://pan.baidu.com/s/18vsSW7eBcP_8ngMQar6fWA?pwd=w53e)
  -[MineRLNavigate-v0](https://pan.baidu.com/s/1KNyvmvtk8YrVPsumA6sRXg?pwd=f2l1)
3. If you encounter difficulties with the above methods, you can download them from [Minerl's official website](https://minerl.readthedocs.io/en/v0.4.4/tutorials/data_sampling.html).
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
