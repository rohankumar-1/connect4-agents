"""
implementing the training / iterative improvement algorithm
"""
from numpy import ndarray
import math
import torch
from torch import Tensor
from tqdm import tqdm
from model import PolicyValueNetwork
from state import Game
from collections import defaultdict
from utils import _save_to_safetensor
from concurrent.futures import ProcessPoolExecutor

MCTS = 50 # paper has 800 iterations for chess (connect4 has smaller space)
C_ULT = 0.5 # tradeoff between eploitation, exploration

class AlphaZero:

    def __init__(self, noise:float=0.0, model_pth=None, train=False):
        self.dirichlet_dist = torch.distributions.Dirichlet(concentration=(torch.ones((7,))*noise))
        self.model = PolicyValueNetwork()
        if model_pth is not None:
            self.model._load_checkpoint(model_pth)
        self.train = train
        if self.train:
            self.data = []

    def _pult(self, s, mask):
        """ predicted upper confidence bound applied to trees """
        raw_pult: Tensor = (self.q[s] / (1e-5 + self.visits[s])) + (C_ULT * self.priors[s]*math.sqrt(self.N) / (1 + self.visits[s]))
        raw_pult[mask] = -torch.inf
        return raw_pult

    def _reset_dicts(self):
        self.N = 0
        self.priors: dict[bytes, Tensor]  = defaultdict()                                                     # probability distribution over moves from s
        self.visits: dict[bytes, Tensor]  = defaultdict(lambda: torch.zeros(size=(7,), dtype=torch.int32))    # times we have gotten to state s
        self.q: dict[bytes, Tensor]       = defaultdict(lambda: torch.zeros(size=(7,), dtype=torch.float32))  # running sum of values seen from state s

    def get_best_move(self, game: Game) -> int:
        """ simulate a bunch of model-guided MCTS, then pick the action that brings us to the most visited state """
        self._reset_dicts()

        # set initial priors from model + Dirichlet noise (at root node, we have no information yet)
        s: bytes = game.get_hash()
        prior_pred, _ = self.model.predict(game.get_state_tensor())
        noise: Tensor = self.dirichlet_dist.sample()
        self.priors[s] = 0.75 * prior_pred + 0.25 * noise
        mask: ndarray = game.get_invalid_moves()

        # do large # of tree searches (via model-guided MCTS)
        for _ in range(MCTS):
            pult: Tensor = self._pult(s, mask)
            move: int = int(torch.argmax(pult).item()) 
            self._value(game, move)

        # get the move that was visited the most (if model is good, visits == strength of move)
        d: Tensor = self.visits[s]
        if self.train:
            self.data.append({"s_t": game.get_state_tensor(), "alpha_t": Tensor(self.visits[s]/MCTS), "turn": game.turn})
        return torch.argmax(d).numpy()

    def get_data(self, game_result: int):
        """ reset data for a new game: if turn is the same as the game winner, then encourage this sample, else discourage """
        res: list = self.data.copy()
        for sample in res:
            if game_result == 0:
                sample['z_t'] = 0.0
            else:
                sample["z_t"] = 1.0 if (sample["turn"] == game_result) else -1.0 
        self.data: list = []
        return res

    def _value(self, game: Game, a: int) -> float:
        s= game.get_hash()          # original state
        game.make_move(a)
        s_prime = game.get_hash()   # new state after move
        if game.over():
            v: float = not game.none_empty()
        elif self.visits[s][a] > 0: 
            next_move: int = int(torch.argmax(self._pult(s, mask=game.get_invalid_moves())).item())
            v: float = -self._value(game, next_move) # recursive call
        else:
            self.priors[s_prime], v = self.model.predict(game.get_state_tensor())

        game.undo_move()
        self.visits[s][a] += 1
        self.q[s][a] += v
        self.N += 1
        return v



def play_single_game(model_path, noise):
    # Each process must load its own local copy of the model
    bot = AlphaZero(noise=noise, model_pth=model_path, train=True)
    game = Game()
    
    while not game.over():
        move = bot.get_best_move(game)
        game.make_move(move)
        
    # The score is usually +1 for the last player to move, -1 for the loser
    return bot.get_data(game.score())

def selfplay_parallel(games=100, noise=0.3, model_path=None, outpath="data/iter001.safetensors", workers=4):
    all_data = []
    
    # Use a ProcessPool to run games in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all game tasks
        futures = [executor.submit(play_single_game, model_path, noise) for _ in range(games)]
        
        for f in tqdm(futures, desc="Self-play Progress"):
            game_data = f.result() # This is the list of (s, p, z) for one game
            all_data.extend(game_data)

    _save_to_safetensor(all_data, outpath)



if __name__=="__main__":
    selfplay_parallel(games=100, noise=0.3, workers=4)