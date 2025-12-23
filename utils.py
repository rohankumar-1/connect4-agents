
import torch
import random
from state import Game


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
