from pandas.core.internals.blocks import new_block_2d

import torch
from torch.utils.data import DataLoader
from zero import AlphaZero
from state import Game
from utils import PolicyValueDataset
from model import PolicyValueNetwork
from argparse import ArgumentParser
import torch.multiprocessing as mp
from queue import Empty
import time
from tqdm import trange, tqdm

parser = ArgumentParser("AlphaZero training")
parser.add_argument("--data", "-d")
parser.add_argument("--model", "-m")

# Use 'spawn' to avoid issues with MPS/GPU contexts on macOS
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def inference_server(model_path, request_queue, response_queues):
    device = torch.device("mps")
    model = PolicyValueNetwork().to(device)
    if model_path and ".safetensors" in model_path:
        model._load_checkpoint(model_path)
    model.eval()

    while True:
        try:
            # Non-blocking check for the "Stop" signal
            request = request_queue.get(timeout=0.1)
            if request == "STOP": break
            
            batch_states, batch_ids = [request[1]], [request[0]]
            
            # 1. Collect a batch (up to 32 for M3 Pro efficiency)
            start_time = time.time()
            while len(batch_states) < 32 and (time.time() - start_time < 0.005):
                try:
                    req = request_queue.get(timeout=0.001)
                    if req == "STOP": 
                        break
                    batch_states.append(req[1])
                    batch_ids.append(req[0])
                except Empty: 
                    break
            
            # 2. Batch Inference
            # Ensure states are (N, 2, 6, 7). Remove squeeze unless you're sure of 5D input
            states_tensor = torch.stack(batch_states).to(device)
            if states_tensor.dim() == 5: states_tensor = states_tensor.squeeze(1)

            with torch.no_grad():
                probs, values = model.predict(states_tensor)
                probs = probs.cpu()
                v_out = values.cpu().numpy()
                
            for i, worker_id in enumerate(batch_ids):
                response_queues[worker_id].put((probs[i], v_out[i].item()))
        except Empty:
            continue

def actor_worker(worker_id, num_games, noise, req_q, res_q, data_q):
    def remote_predict(state_tensor):
        req_q.put((worker_id, state_tensor))
        return res_q.get()

    bot = AlphaZero(noise=noise, train=True)
    bot.model.predict = remote_predict 

    for _ in range(num_games):
        game = Game()
        while not game.over():
            move = bot.get_best_move(game)
            game.make_move(move)
        data_q.put(bot.get_data(game.score()))
    
    data_q.put("WORKER_DONE") # Signal this worker is finished

def selfplay_parallel(games=100, model_path=None, workers=6):
    req_q = mp.Queue()
    data_q = mp.Queue()
    res_qs = [mp.Queue() for _ in range(workers)]
    
    server = mp.Process(target=inference_server, args=(model_path, req_q, res_qs))
    server.start()
    
    actors = [mp.Process(target=actor_worker, args=(i, games//workers, 0.3, req_q, res_qs[i], data_q)) for i in range(workers)]
    for p in actors: p.start()
    
    all_data = []
    finished_workers = 0
    pbar = tqdm(total=games, desc="Self-play")
    
    while finished_workers < workers:
        msg = data_q.get()
        if msg == "WORKER_DONE":
            finished_workers += 1
        else:
            all_data.extend(msg)
            pbar.update(1)
    
    pbar.close()
    
    # Graceful Shutdown
    req_q.put("STOP")
    for p in actors: p.join()
    server.join()
    
    return all_data
if __name__=="__main__":
    args = parser.parse_args()      

    data_path = f"data/iter{int(args.data):03}.safetensors"
    old_model_path = f"models/iter{int(args.model)-1:03}.safetensors"
    new_model_path = f"models/iter{int(args.model):03}.safetensors"

    # SELFPLAY
    selfplay_parallel(games=100, model_path=old_model_path, workers=6)

    # RETRAIN
    dataset = PolicyValueDataset()
    print("Samples in dataset:",  len(dataset))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    net = PolicyValueNetwork()
    net._load_checkpoint(old_model_path)
    optimizer = torch.optim.AdamW(params=net.parameters())
    net.train_iteration(train_loader, optimizer=optimizer, outpath=new_model_path)