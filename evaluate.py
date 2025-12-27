"""
With the most recent model, how far off from the ground-truth best moves are we?
"""
import pandas as pd
from model import PolicyValueNetwork
from ucimlrepo import fetch_ucirepo
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import torch

# fetch dataset 
connect_4 = fetch_ucirepo(id=26) 
X: pd.DataFrame = connect_4.data.features 
y: pd.Series = connect_4.data.targets 
parser = ArgumentParser("evaluation")
parser.add_argument("--model_id", "-mid")

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

def evaluate(net, X, y, batch_size=512):
    net.eval()
    device = next(net.parameters()).device
    correct = 0
    total = len(y)
    
    # Process in large batches
    for i in tqdm(range(0, total, batch_size), desc="UCI Eval"):
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

if __name__ == "__main__":
    # Example usage for a standalone test
    net = PolicyValueNetwork("models/best001.safetensors")
    accuracy = evaluate(net, X, y, batch_size=64)
    print(f"Value Head Accuracy: {accuracy:.2%}")