import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_loaders

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
    def __init__(self, num_res_blocks=4, num_channels=64, path=None):
        super().__init__()

        self.start_block = nn.Sequential(
            nn.Conv2d(2, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.res_tower = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])
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
            nn.Linear(1*6*7, 1),
            nn.Tanh() # to map to [-1, 1]
        )
        if path is not None:
            self._load_checkpoint(path)

    def forward(self, x: torch.Tensor):
        x = self.start_block(x)
        x = self.res_tower(x)
        policy_logits = self.policy_head(x) # Output: [Batch, 7]
        value = self.value_head(x)          # Output: [Batch, 1]
        return policy_logits, value

    def predict(self, x):
        """ helper for MCTS inference """
        with torch.no_grad():
            policy_logits, values = self.forward(x.unsqueeze(0))
            probs = torch.softmax(policy_logits, dim=1)
        return probs.squeeze(0), values

    def batch_predict(self, x, softmax=True):
        with torch.no_grad():
            policy_logits, values = self.forward(x)
            if softmax:
                return torch.softmax(policy_logits, dim=1), values
            else:
                return policy_logits, values


    def _save_checkpoint(self, path):
        """ helper, saves model state to path """
        torch.save({'model_state_dict': self.state_dict()}, path)
        print(f"Saved trained model to {path}")

    def _load_checkpoint(self, path):
        t = torch.load(path, weights_only=True)
        self.load_state_dict(t['model_state_dict'])

    def train_epoch(self, trainloader, optimizer, epoch_num):
        self.train()
        epoch_loss = 0.0
        for (s, p, z) in trainloader:
            optimizer.zero_grad()
            pred_policy, pred_value = self.forward(s)
            
            total_loss = policy_value_loss(p, z, pred_policy, pred_value)
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.detach().item()

        return epoch_loss / len(trainloader)

    def validate(self, testloader):
        self.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for (s,p,z) in testloader:
                pred_policy, pred_value = self.batch_predict(s)
                loss = policy_value_loss(p,z, pred_policy, pred_value)
                probs = torch.softmax(pred_policy, dim=1)
                val_loss += loss.detach().item()
                val_acc += (torch.argmax(probs, dim=1) == torch.argmax(p, dim=1)).sum()
        return val_loss / len(testloader), val_acc / len(testloader)


def policy_value_loss(p, z, policy, value, weight=1.0):
    """ build policy-value loss: L = (z - v)^2 - pi^T * log(p) + R(\theta), regularization from optimizer """
    value_loss: torch.Tensor = F.mse_loss(value.view(-1), z.view(-1))
    policy_loss: torch.Tensor = F.kl_div(F.log_softmax(policy, dim=1), p, reduction='batchmean')
    return value_loss + weight * policy_loss


if __name__=="__main__":
    trainloader, testloader = get_loaders(window=10, batch_size=32, test_split=0.1)
    net = PolicyValueNetwork()
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=1e-4, weight_decay=5e-5)
    
    for epoch in range(25):
        train_loss = net.train_epoch(trainloader, optimizer, epoch)
        val_loss, val_acc = net.validate(testloader)
        print(f"Epoch {epoch:02d}: train loss = {train_loss:6.4f} | val loss = {val_loss:8.4f} | val acc = {val_acc:.4f}")

    net._save_checkpoint("models/best001.safetensors")