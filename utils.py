
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import trange
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd


def _save_to_safetensor(data:list, path:str):
    state_tensors = []
    policy_tensors = []
    value_tensors = []

    for sample in data:
        state_tensors.append(sample['s_t'].squeeze())
        policy_tensors.append(sample['alpha_t'])
        value_tensors.append(sample['z_t'])

    full= {
        "states": torch.stack(state_tensors),        # Shape: [N, 3, 6, 7] (3 channels, one for X, one for O, one for turn)
        "policies": torch.stack(policy_tensors),     # Shape: [N, 7] (7 possible moves)
        "values": torch.tensor(value_tensors)           # Shape: [N, 1]
    }

    torch.save(full, path)

class PolicyValueDataset(Dataset):
    def __init__(self, states, policies, values, augment=False):
        self.states = states
        self.policies = policies
        self.values = values
        self.augment = augment

    @classmethod
    def from_disk(cls, window=10, data_dir="./../data/", augment=False):
        raw_states, raw_policies, raw_values = [], [], []
        
        # Filter for safetensors or pt files
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.safetensors') or f.endswith('.pt')])[-window:]
        if not files:
            raise FileNotFoundError(f"No data files found in {data_dir}")
            
        print(f"Aggregating {len(files)} files...")
        for file in files:
            # Note: ensure you use the correct loader for your file format
            # If using safetensors, use safetensors.torch.load_file
            d = torch.load(os.path.join(data_dir, file), weights_only=True)
            raw_states.append(d['states'].view(-1, 2, 6, 7)) # Force standard shape
            raw_policies.append(d['policies'].view(-1, 7))
            raw_values.append(d['values'].view(-1))

        all_states = torch.cat(raw_states, dim=0)
        all_policies = torch.cat(raw_policies, dim=0)
        all_values = torch.cat(raw_values, dim=0)

        # State Averaging Logic
        aggregated_data = {}
        print("Averaging duplicate states...")
        for i in range(len(all_states)):
            s_hash = all_states[i].numpy().tobytes()
            if s_hash not in aggregated_data:
                aggregated_data[s_hash] = [all_states[i], all_policies[i], all_values[i], 1]
            else:
                aggregated_data[s_hash][1] += all_policies[i]
                aggregated_data[s_hash][2] += all_values[i]
                aggregated_data[s_hash][3] += 1

        final_s, final_p, final_v = [], [], []
        for s, p_sum, v_sum, count in aggregated_data.values():
            final_s.append(s.unsqueeze(0))
            final_p.append((p_sum / count).unsqueeze(0))
            final_v.append(torch.tensor([v_sum / count], dtype=torch.float32))

        print(f"Compressed {len(all_states)} -> {len(final_s)} unique states.")
        return cls(torch.cat(final_s), torch.cat(final_p), torch.cat(final_v), augment=augment)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        state = self.states[idx].clone()
        policy = self.policies[idx].clone()
        value = self.values[idx].clone()

        if self.augment and torch.rand(1) < 0.5:
            state = torch.flip(state, dims=[2]) 
            policy = torch.flip(policy, dims=[0]) 

        return state, policy, value

def get_loaders(window=10, batch_size=32, test_split=0.1):
    # 1. Load and Average everything
    full_ds = PolicyValueDataset.from_disk(window=window)
    
    # 2. Split Tensors manually to create two distinct Dataset objects
    total_indices = torch.randperm(len(full_ds))
    test_size = int(len(full_ds) * test_split)
    
    test_idx = total_indices[:test_size]
    train_idx = total_indices[test_size:]
    
    train_dataset = PolicyValueDataset(
        full_ds.states[train_idx], 
        full_ds.policies[train_idx], 
        full_ds.values[train_idx], 
        augment=True # Enable flipping for training
    )
    
    test_dataset = PolicyValueDataset(
        full_ds.states[test_idx], 
        full_ds.policies[test_idx], 
        full_ds.values[test_idx], 
        augment=False # Disable flipping for clean validation
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


#########################################################################
# 
#   Evaluation helpers  
#
######################################################################### 

def _get_uci_dataset():
    connect_4 = fetch_ucirepo(id=26) 
    X: pd.DataFrame = connect_4.data.features 
    y: pd.Series = connect_4.data.targets 
    return X, y

def get_outcome_map(x: str): 
    return {'win': 1, 'loss': -1, 'draw': 0}.get(x, 0)

def build_batch_tensors(X_subset, y_subset):
    """ convert UCI format to games """
    mapping = {'x': 1, 'b': 0, 'o': -1}
    raw_data = X_subset.replace(mapping).values.astype(np.float32) # (Batch, 42)
    boards = raw_data.reshape(-1, 7, 6).transpose(0, 2, 1)
    boards = np.flip(boards, axis=1).copy()
    p1_chan = (boards == 1).astype(np.float32)
    p2_chan = (boards == -1).astype(np.float32)
    states = np.stack([p1_chan, p2_chan], axis=1) # (Batch, 2, 6, 7)
    outcomes = np.array([get_outcome_map(val[0]) for val in y_subset.values])
    return torch.from_numpy(states), torch.from_numpy(outcomes)

def evaluate(net, batch_size=128, verbose=True):
    X, y= _get_uci_dataset()
    net.eval()
    device = next(net.parameters()).device
    correct = 0
    total = len(y)
    
    # Process in large batches
    range_fn = trange if verbose else range
    for i in range_fn(0, total, batch_size):
        X_batch = X.iloc[i : i + batch_size]
        y_batch = y.iloc[i : i + batch_size]
        
        # 1. Prepare Tensors
        states, targets = build_batch_tensors(X_batch, y_batch)
        states = states.to(device)
        
        # 2. Batch Inference
        with torch.no_grad():
            _, v = net.predict(states) # v shape: (Batch, 1)
            v = v.cpu().view(-1)
            
        # 3. Categorize
        pred_class = torch.zeros_like(v)
        pred_class[v > 0.1] = 1
        pred_class[v < -0.1] = -1
        
        correct += (pred_class == targets).sum().item()

    return correct / total