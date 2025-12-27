
import torch
from torch.utils.data import Dataset, DataLoader
import random
from state import Game
import os


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
    """ 
    The PolicyValueDataset properly formats data samples, performing the following functions: 
    1) Load in the most recent 4 selfplay iterations: games later than that probably use really weak models, so we can safely ignore 
    2) Average out p,v for states that were seen multiple times
    3) Applies data augmentations (flipping state/policy vectors, since connect-four is symmetric)
    4) Returns single samples for Dataloader
    """

    def __init__(self, window=10, data_dir:str="./data/"):
        raw_states = []
        raw_policies = []
        raw_values = []
        
        # 1. Load the raw data from the window
        files = sorted([f for f in os.listdir(data_dir)])[-window:]
        print(f"Aggregating {len(files)} files for training...")
        
        for file in files:
            d = torch.load(os.path.join(data_dir, file), weights_only=True)
            raw_states.append(d['states'].squeeze())
            raw_policies.append(d['policies'])
            raw_values.append(d['values'].view(-1)) # Ensure 1D

        # Combine all loaded data
        all_states = torch.cat(raw_states, dim=0)
        all_policies = torch.cat(raw_policies, dim=0)
        all_values = torch.cat(raw_values, dim=0)

        # 2. State Averaging Logic
        aggregated_data = {}
        
        print("Averaging duplicate states...")
        for i in range(len(all_states)):
            # Convert state to a hashable tuple or byte-string
            # Using .numpy().tobytes() is usually the fastest way to hash a tensor
            s_hash = all_states[i].numpy().tobytes()
            
            if s_hash not in aggregated_data:
                aggregated_data[s_hash] = {
                    'state': all_states[i],
                    'policy_sum': all_policies[i],
                    'value_sum': all_values[i],
                    'count': 1
                }
            else:
                aggregated_data[s_hash]['policy_sum'] += all_policies[i]
                aggregated_data[s_hash]['value_sum'] += all_values[i]
                aggregated_data[s_hash]['count'] += 1

        # 3. Final Tensor Construction
        self.states = []
        self.policies = []
        self.values = []

        for entry in aggregated_data.values():
            count = entry['count']
            self.states.append(entry['state'].unsqueeze(0))
            # Average the policy and value
            self.policies.append((entry['policy_sum'] / count).unsqueeze(0))
            self.values.append(torch.tensor([entry['value_sum'] / count]))

        self.states = torch.cat(self.states, dim=0)
        self.policies = torch.cat(self.policies, dim=0)
        self.values = torch.cat(self.values, dim=0)
        
        print(f"Dataset complete: {len(all_states)} raw samples compressed to {len(self.states)} unique states.")
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        state = self.states[idx]
        policy = self.policies[idx]
        value = self.values[idx]
        if torch.rand(1) < 0.5:
            state = torch.flip(state, dims=[2]) # Flip columns
            policy = torch.flip(policy, dims=[0]) # Flip move probabilities
        return state, policy, value



class SimpleBot:
    def get_best_move(self, game: Game):
        return random.choice(game.get_valid_moves())


class LookaheadBot:
    """ looks ahead one time """
    def get_best_move(self, game):

        for move in game.get_valid_moves():
            game.make_move(move)
            if game.over():
                game.undo_move()
                return move
            game.undo_move()

        return random.choice(game.get_valid_moves())
