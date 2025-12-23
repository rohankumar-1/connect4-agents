import hashlib
import torch
import random
from state import Game


def hash_tensor(t: torch.Tensor) -> bytes:
    """ 
    hash the state representation, returning byte string. this implementation is not shape/dtype sensitive, 
    but we use the same shape/dtype for every state rep in this project
    """
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).digest()



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
