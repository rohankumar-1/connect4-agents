from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PolicyValueDataset
from tqdm import tqdm

class ResBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class PolicyValueNetwork(nn.Module):
    def __init__(self, num_res_blocks=4, num_channels=64):
        super().__init__()

        # Channel 0: Current Player Pieces, Channel 1: Opponent Pieces
        self.start_block = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # deep residual tower
        self.res_tower = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            # Output from Conv: [Batch, 2, 6, 7]
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7) 
        )
    
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1), # output is [1, 6, 7]
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # to map to [-1, 1]
        )

    def forward(self, x: torch.Tensor):
        x = self.start_block(x)
        
        for block in self.res_tower:
            x = block(x)
        
        policy_logits = self.policy_head(x) # Output: [Batch, 7]
        value = self.value_head(x)          # Output: [Batch, 1]
        
        return policy_logits, value

    def predict(self, x):
        """ helper for MCTS inference """
        self.eval()
        with torch.no_grad():
            # If single board passed, add batch dimension
            if x.ndimension() == 3:
                x = x.unsqueeze(0)
                
            policy_logits, value = self.forward(x)
            probs = torch.softmax(policy_logits, dim=1)
            
        return probs.squeeze(0).cpu(), value.item()


    def _save_checkpoint(self, path):
        """ helper, saves model state to path """
        torch.save({
            'model_state_dict': self.state_dict()
        }, path)
        print(f"Saved trained model to {path}")

    def _load_checkpoint(self, path):
        t = torch.load(path, weights_only=True)
        self.load_state_dict(t['model_state_dict'])


    def train_iteration(self, trainloader: DataLoader, optimizer, outpath:str, epochs:int=10):
        """ trains model, and saves the new model to outpath """
        self.train()
        for epoch in range(epochs):
            
            for (s, p, z) in tqdm(trainloader, desc=f"Epoch {epoch}:", total=len(trainloader)):
                optimizer.zero_grad()
                pred_policy, pred_value = self.forward(s)

                # build policy-value loss: L = (z - v)^2 - pi^T * log(p) + R(\theta)
                # weight decay comes from AdamW
                value_loss: torch.Tensor = F.mse_loss(pred_value, z)
                policy_loss: torch.Tensor = -torch.sum(p * F.log_softmax(pred_policy, dim=1)) / s.size(0)
                total_loss = value_loss + policy_loss

                total_loss.backward()
                optimizer.step()
            
        self._save_checkpoint(outpath)


if __name__=="__main__":
    dataset = PolicyValueDataset()
    print("Samples in dataset:",  len(dataset))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    net = PolicyValueNetwork()
    optimizer = torch.optim.AdamW(params=net.parameters())
    net.train_iteration(train_loader, optimizer=optimizer, outpath="models/iter001.safetensors")