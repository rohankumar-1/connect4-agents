import torch
from zero import AlphaZero
from state import Game
from utils import get_loaders, _save_to_safetensor
from model import PolicyValueNetwork
from argparse import ArgumentParser
from tqdm import trange
from ucimlrepo import fetch_ucirepo
from evaluate import evaluate

parser = ArgumentParser("AlphaZero training")
parser.add_argument("--iter", "-i")
parser.add_argument("--start", "-s")

def run_sequential_selfplay(num_games, model_path, noise=0.3):
    """Runs games one by one on a single process."""
    bot = AlphaZero(noise=noise, MCTS=600, C_PUCT=1.1, model_pth=model_path, train=True, random_select=True)
    
    all_data = []
    for _ in trange(num_games):
        game = Game()
        while not game.over():
            move = bot.get_best_move(game)
            game.make_move(move)
        
        game_data = bot.get_data(game.score())
        all_data.extend(game_data)
        
    return all_data


if __name__ == "__main__":
    parser = ArgumentParser("AlphaZero Sequential Training")
    parser.add_argument("--iter", "-i", type=int, default=1, help="Number of iterations to run")
    parser.add_argument("--start", "-s", type=int, default=1, help="Starting iteration index")
    args = parser.parse_args()

    # Load UCI dataset once
    print("Loading UCI Benchmark Dataset...")
    connect_4 = fetch_ucirepo(id=26) 
    X, y = connect_4.data.features, connect_4.data.targets 

    for i in range(args.start, args.start + args.iter):
        old_idx = i - 1
        data_path = f"data/iter{i:03}.safetensors"
        old_model_path = f"models/iter{old_idx:03}.safetensors" if old_idx != 0 else "models/start001.safetensors"
        new_model_path = f"models/iter{i:03}.safetensors"

        print(f"\n{'='*20} ITERATION {i:03d} {'='*20}")
        all_data = run_sequential_selfplay(num_games=100, model_path=old_model_path)
        _save_to_safetensor(all_data, data_path)

        train_loader, test_loader = get_loaders(window=10, batch_size=64, test_split=0.1)
        
        net = PolicyValueNetwork(path=old_model_path)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=5e-5)
        
        print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Acc':<10}")
        print("-" * 50)

        best_val_loss = float('inf')
        for epoch in range(1, 26): # 25 Epochs
            t_loss = net.train_epoch(train_loader, optimizer, epoch)
            v_loss, v_acc = net.validate(test_loader)
            
            # Print formatted row
            print(f"{epoch:<8d} | {t_loss:<12.4f} | {v_loss:<12.4f} | {v_acc:<12.4f}")

            # Optional: Save 'best' model based on validation loss during this iteration
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                net._save_checkpoint(new_model_path)

        try:
            # We evaluate the newly trained model against perfect-play data
            uci_acc = evaluate(net, X, y)
            print(f"\n--- UCI Benchmark Accuracy: {uci_acc:.2%} ---")
        except Exception as e:
            print(f"\n[!] UCI Evaluation failed: {e}")
