import numpy as np
import os
from pathlib import Path
from random import sample

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from CDDPM2 import ActionGeneratorCNN, MineRLDataset, ActionGaussianDiffusion, TimeEmbedding, default, encode_action

def extract_features(diffusion_model, data_loader, device):
    """Extract features from the model using the given DataLoader."""
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    extracted_features = []
    with torch.no_grad():
        for images, actions in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            actions = actions.to(device)
            timesteps = torch.full((images.size(0),), diffusion_model.num_timesteps - 1, device=device)
            features = diffusion_model.model(images, actions, timesteps).detach().cpu().numpy()
            extracted_features.append(features)
    return np.concatenate(extracted_features, axis=0)

def plot_tsne(features, output_dir, perplexity=30, n_components=2, init='pca', random_state=23):
    """Plot t-SNE projection of the features and save the plot."""
    tsne = TSNE(perplexity=perplexity, n_components=n_components, init=init, random_state=random_state)
    transformed_features = tsne.fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(transformed_features[:, 0], transformed_features[:, 1], marker='.', alpha=0.7)
    plt.title('t-SNE Projection of Image Features')
    plt.savefig(Path(output_dir) / 'tsne_projection.png')
    plt.close()

def plot_pca(features, output_dir, n_components=2):
    """Plot PCA projection of the features and save the plot."""
    pca = PCA(n_components=n_components)
    transformed_features = pca.fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(transformed_features[:, 0], transformed_features[:, 1], marker='.', alpha=0.7)
    plt.title('PCA Projection of Image Features')
    plt.savefig(Path(output_dir) / 'pca_projection.png')
    plt.close()

os.environ['MINERL_DATA_ROOT'] = '/root/autodl-tmp'
dataset = MineRLDataset()
n_samples = 500

# Randomly select n images from the original dataset
indices = sample(range(len(dataset)), n_samples)
new_dataset = Subset(dataset, indices)

data_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = '/root/CDDPM/results7/model_best.pth'
checkpoint = torch.load(model_path, map_location=device)
model = ActionGeneratorCNN(dim=128, dim_mults=(1, 2, 4, 8), channels=3, action_dim=11).to(device)
diffusion_model = ActionGaussianDiffusion(model=model, image_size=64, shape=(1, 11), timesteps=1000,
                                          sampling_timesteps=250, objective='pred_x0').to(device)
diffusion_model.load_state_dict(checkpoint['model_state_dict'])

features = []
for img, _ in data_loader:
    img_features = diffusion_model.extract_image_features(img.to(device))
    features.append(img_features.cpu().detach().numpy())
features = np.concatenate(features, axis=0)

output_dir = '/root/CDDPM/results'
# Analyze image features using t-SNE and PCA
plot_tsne(features, output_dir)
plot_pca(features, output_dir)
