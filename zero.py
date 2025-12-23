"""
implementing the training / iterative improvement algorithm
"""
import math
from sympy import S
import torch
from model import BasicNet
from state import Game
from utils import hash_tensor
from collections import defaultdict

MCTS = 800 # same as paper
C_ULT = 0.3 # tradeoff between eploitation, exploration


class AlphaZero:

    def __init__(self):
        self.model = BasicNet()

    def _pult(self, s, mask):
        return (self.q[s] / self.visits[s]) + C_ULT * (self.priors[s]*mask)*(1 / (1 + self.visits[s]))

    def get_best_move(self, game: Game) -> int:
        """ simulate a bunch of model-guided MCTS, then pick the action that brings us to the most visited state """
        self.priors: dict[bytes, torch.Tensor]  = dict()                        # probability distribution over moves from s
        self.visits: dict[bytes, torch.Tensor]  = dict(torch.zeros(size=(7,), dtype=torch.int32))   # times we have gotten to state s
        self.q: dict[bytes, torch.Tensor]       = dict(torch.zeros(size=(7,), dtype=torch.float32))        # running sum of values seen from state s

        # set initial priors from model (at root node, we have no information yet)
        h = hash_tensor(game.get_state_tensor())
        self.priors[h], _ = self.model(game)

        for _ in range(MCTS):
            self.simulate(game)

        # get the move that was visited the most (if model is good, == strength of move)
        d = self.visits[hash_tensor(game.get_state_tensor())]
        return torch.argmax(d).numpy()

    def simulate(self, game: Game) -> None:
        """ this instantiates a single MCTS iteration. Recurse when possible, in order to find value for each path """

        def value(game: Game, a: int) -> float:
            s= hash_tensor(game.get_state_tensor())
            if self.visits[s][a] > 0: 
                mask = torch.zeros((7,)) # mask probability of illegal moves to 0
                mask[game.get_valid_moves()] = 1.
                next_move = torch.argmax(self._pult(s, mask)).numpy()
                game.make_move(next_move)
                v = -value(game, next_move)
                game.undo_move()
            elif game.over():
                v = game.score()
            else:
                priors, v = self.model(game.get_state_tensor())
                self.priors[s] = priors

            self.visits[s][a] += 1
            return v
        
        # initial node: use PULT with noised priors (avoid lopsided tree exploration, argmax favors earlier idx)
        s = hash_tensor(game.get_state_tensor())
        mask = torch.zeros((7,))
        mask[game.get_valid_moves()] = 1.
        torch.argmax(self._pult(s, mask)).numpy()





