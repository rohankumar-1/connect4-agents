from agents.alphazero import AlphaZeroAgent
from argparse import ArgumentParser


parser = ArgumentParser("AlphaZero training")
parser.add_argument("--verbose", "-v", action='store_true')
parser.add_argument("--MCTS", "-m", type=int, default=400, help="number of MCTS simulations")
parser.add_argument("--iterations", "-i", type=int, default=1, help="number of iterations of selfplay + net training to perform")
parser.add_argument("--start", "-s", type=int, default=1, help="iteration to start training from. Will find model named 'weights/iter___.safetensors'")

if __name__ == "__main__":
    args = parser.parse_args()
    
    for i in range(args.start, args.start + args.iterations):
        old_idx = i - 1
        data_path = f"data/iter{i:03}.safetensors"
        old_model_path = f"models/weights/iter{old_idx:03}.safetensors" if old_idx != 0 else "models/start001.safetensors"
        new_model_path = f"models/weights/iter{i:03}.safetensors"

        if args.verbose:
            print("#"*50)
            print(f"#   ITERATION {i}") 
            print("#"*50)

        bot = AlphaZeroAgent(noise=0.3, MCTS=int(args.MCTS), C_PUCT=1.1, model_path=old_model_path, train=True, random_select=True)
        bot.train_iteration(data_out_path=data_path, model_out_path=new_model_path, net_train_epochs=15, verbose=args.verbose, eval=True)

