import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import minerl

def encode_action(action):
    discrete_actions = np.array([
        action['attack'], action['back'], action['forward'],
        action['jump'], action['left'], action['right'],
        action['sneak'], action['sprint']
    ]).flatten()

    camera_actions = np.array(action['camera']).flatten() / 180.0

    place_action = np.array([1]) if action['place'] == 'dirt' else np.array([0])

    return np.concatenate([discrete_actions, camera_actions, place_action])

class MineRLDataset(Dataset):
    def __init__(self, environment='MineRLNavigate-v0', transform=None, img_size=128, max_samples=20000):
        self.img_size = img_size
        self.data = minerl.data.make(environment)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.samples = []
        iterator = minerl.data.BufferedBatchIter(self.data)
        for current_state, action, _, _, _ in iterator.buffered_batch_iter(batch_size=1, num_epochs=1):
            if len(self.samples) >= max_samples:
                break
            pov_image = current_state['pov'][0]
            encoded_action = encode_action(action)
            self.samples.append((pov_image, encoded_action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pov_image, encoded_action = self.samples[idx]
        pov_image = self.transform(pov_image)
        encoded_action = torch.tensor(encoded_action, dtype=torch.float32)
        return pov_image, encoded_action

class ActionCNN(nn.Module):
    def __init__(self, channels=3, num_discrete_actions=9, num_continuous_actions=2, dim=16, dim_mults=(1, 2, 4, 8), pool=5):
        super().__init__()
        # Initialize dimensions
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        
        # Convolutional layers
        self.features = nn.Sequential()
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            self.features.add_module(f"conv{i}", nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1))
            self.features.add_module(f"batchnorm{i}", nn.BatchNorm2d(out_dim))
            self.features.add_module(f"relu{i}", nn.ReLU(inplace=True))

        # Adaptive pooling to reduce to a fixed size output irrespective of input size
        self.global_pool = nn.AdaptiveAvgPool2d((pool, pool))
        
        # Final fully connected layers for different types of actions
        self.fc_discrete = nn.Linear(out_dim * pool * pool, num_discrete_actions)
        self.fc_continuous = nn.Linear(out_dim * pool * pool, num_continuous_actions)

    def forward(self, x):
        # Image features
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Output layers for discrete and continuous actions
        out_discrete = torch.sigmoid(self.fc_discrete(x))  # Sigmoid activation for probabilities
        out_continuous = self.fc_continuous(x)  # No activation for continuous values

        return out_discrete, out_continuous
    def extract_features(self, img):
        # Only the image features are processed and returned
        img_features = self.features(img)
        img_features = self.global_pool(img_features)
        img_features = torch.flatten(img_features, 1)
        return img_features
    
def compute_loss(outputs_discrete, outputs_continuous, labels_discrete, labels_continuous):
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    loss_discrete = criterion_bce(outputs_discrete, labels_discrete)
    loss_continuous = criterion_mse(outputs_continuous, labels_continuous)
    return 0.1 * loss_discrete + 0.9 * loss_continuous

def train_and_evaluate_model(model, train_loader, test_loader, optimizer, num_epochs, eval_interval, results_folder, device):
    best_loss = float('inf')
    log_file_path = os.path.join(results_folder, 'training_log.txt')
    
    # Open log file and write headers
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch,Phase,Loss,Best Test Loss\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        # Training phase
        for images, actions in train_loader:
            images, actions = images.to(device), actions.to(device)
            optimizer.zero_grad()
            outputs_discrete, outputs_continuous = model(images)
            actions_discrete, actions_continuous = actions[:, :9].float(), actions[:, 9:11].float()
            loss = compute_loss(outputs_discrete, outputs_continuous, actions_discrete, actions_continuous)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # Record training loss
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch + 1},Train,{avg_train_loss:.4f},{best_loss:.4f}\n")
        
        # Testing phase at intervals
        if (epoch) % eval_interval == 0:
            avg_test_loss = evaluate_model(model, test_loader, device)
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), os.path.join(results_folder, 'best_model.pth'))
                print(f'New best model saved at epoch {epoch} with test loss: {avg_test_loss:.4f}')
            
            # Record test loss and best loss
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{epoch},Test,{avg_test_loss:.4f},{best_loss:.4f}\n")
            
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, Best Loss = {best_loss:.4f}')
        else:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}')

            
def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, actions in test_loader:
            images, actions = images.to(device), actions.to(device)
            outputs_discrete, outputs_continuous = model(images)
            actions_discrete, actions_continuous = actions[:, :9].float(), actions[:, 9:11].float()
            loss = compute_loss(outputs_discrete, outputs_continuous, actions_discrete, actions_continuous)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

def main():
    os.environ['MINERL_DATA_ROOT'] = '/root/autodl-tmp'
    full_dataset = MineRLDataset()
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    results_folder = '/root/CDDPM/results_regre3'  # Results folder path
    os.makedirs(results_folder, exist_ok=True)

    num_epochs = 500  # Total number of epochs to train
    eval_interval = 1  # Evaluate and save the model every 5 epochs

    train_and_evaluate_model(model, train_loader, test_loader, optimizer, num_epochs, eval_interval, results_folder, device)

if __name__ == '__main__':
    main()
