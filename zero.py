"""
implementing the training / iterative improvement algorithm
"""
from typing import Union
import math
import torch
from model import BasicNet
from state import Game
from collections import defaultdict

MCTS = 50 # same as paper
C_ULT = 0.3 # tradeoff between eploitation, exploration

class AlphaZero:

    def __init__(self, data_dict=None, game_idx=None):
        self.model = BasicNet()
        self.data_dict: Union[dict | None] = data_dict
        self.game_idx: Union[int | None] = game_idx

    def _pult(self, s, mask):
        """ predicted upper confidence bound applied to trees """
        raw_pult =(self.q[s] / (1e-5 + self.visits[s])) + C_ULT * self.priors[s]*(math.sqrt(self.N) / (1 + self.visits[s]))
        raw_pult[mask] = -torch.inf
        return raw_pult

    def _reset_dicts(self):
        self.N = 0
        self.priors: dict[bytes, torch.Tensor]  = defaultdict()                                                     # probability distribution over moves from s
        self.visits: dict[bytes, torch.Tensor]  = defaultdict(lambda: torch.zeros(size=(7,), dtype=torch.int32))    # times we have gotten to state s
        self.q: dict[bytes, torch.Tensor]       = defaultdict(lambda: torch.zeros(size=(7,), dtype=torch.float32))  # running sum of values seen from state s

    def get_best_move(self, game: Game) -> int:
        """ simulate a bunch of model-guided MCTS, then pick the action that brings us to the most visited state """
        self._reset_dicts()

        # set initial priors from model (at root node, we have no information yet)
        s: bytes = game.get_hash()
        self.priors[s], _ = self.model(game)
        mask = game.get_invalid_moves()

        # do large # of tree searches (via model-guided MCTS)
        for _ in range(MCTS):
            # print("STARTING SEARCH")
            pult = self._pult(s, mask)
            # print(f"initial pult: {pult}")
            move = torch.argmax(pult).numpy() 
            self._value(game, move)

        # get the move that was visited the most (if model is good, visits == strength of move)
        d = self.visits[s]
        return torch.argmax(d).numpy()


    def _value(self, game: Game, a: int) -> float:
        s= game.get_hash()          # original state
        # print(game.move_stack)
        # print("==============================")
        # print(f"move: {a}")
        # print("First state: ", game)
        # print(f"past visits to move {a}: {self.visits[s][a]}")
        # print(f"value of move {a}: {self.q[s][a]}")
        # print(f"priors of first state {a}: {self.priors[s]}")
        # print("==============================")
        game.make_move(a)
        s_prime = game.get_hash()   # new state after move
        if game.over():
            # print(" ---> GAME OVER")
            v=  -game.score()
        elif self.visits[s][a] > 0: 
            # print(" ---> GOING TO NEXT NODE")
            move_mask = game.get_invalid_moves()
            # print(f"move mask: {move_mask}")
            next_move = torch.argmax(self._pult(s, move_mask)).numpy()
            v = -self._value(game, next_move)
        else:
            # print(" ---> LEAF NODE HIT, RETREATING")
            self.priors[s_prime], v = self.model(game.get_state_tensor())

        game.undo_move()
        self.visits[s][a] += 1
        self.q[s][a] += v
        self.N += 1
        return v





