"""
implementing the training / iterative improvement algorithm
"""
from numpy import ndarray
import math
import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np
from model import PolicyValueNetwork
from state import Game
from collections import defaultdict
from utils import _save_to_safetensor
from concurrent.futures import ProcessPoolExecutor

MCTS = 400 # paper has 800 iterations for chess (connect4 has smaller space)
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
        raw_pult: Tensor = (self.q[s] / (1e-5 + self.visits[s])) + (C_ULT * self.priors[s]*(math.sqrt(self.visits[s].sum()) / (1 + self.visits[s])))
        raw_pult[mask] = -torch.inf
        return raw_pult

    def _reset_dicts(self):
        self.priors: dict[bytes, Tensor]  = defaultdict()                                                     # probability distribution over moves from s
        self.visits: dict[bytes, Tensor]  = defaultdict(lambda: torch.zeros(size=(7,), dtype=torch.int32))    # times we have gotten to state s
        self.q: dict[bytes, Tensor]       = defaultdict(lambda: torch.zeros(size=(7,), dtype=torch.float32))  # running sum of values seen from state s

    def get_best_move(self, game: Game) -> int:
        """ simulate a bunch of model-guided MCTS, then pick the action that brings us to the most visited state """
        self._reset_dicts()

        # set initial priors from model + Dirichlet noise (at root node, we have no information yet)
        s: bytes = game.get_hash()
        prior_pred, _ = self.model.predict(game.get_state_tensor())
        # print(prior_pred)
        if isinstance(prior_pred, np.ndarray):
            prior_pred = torch.from_numpy(prior_pred)
        noise: Tensor = self.dirichlet_dist.sample()
        self.priors[s] = 0.75 * prior_pred + 0.25 * noise
        mask: ndarray = game.get_invalid_moves()

        # do large # of tree searches (via model-guided MCTS)
        for _ in range(MCTS):
            pult: Tensor = self._pult(s, mask)
            move: int = int(torch.argmax(pult).item()) 
            self._value(game.clone(), move)

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
        elif s_prime in self.priors: 
            next_move: int = int(torch.argmax(self._pult(s, mask=game.get_invalid_moves())).item())
            v: float = -self._value(game, next_move) # recursive call
        else:
            self.priors[s_prime], v = self.model.predict(game.get_state_tensor())
            v = -v # we predicted how s_prime will look (i.e. for the next player), so negate for this one

        game.undo_move()
        self.visits[s][a] += 1
        self.q[s][a] += v
        return v



if __name__=="__main__":
    selfplay_parallel(games=100, noise=0.3, workers=4)