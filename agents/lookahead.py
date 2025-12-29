from state import Game
import random
from .agent import Agent

class LookaheadAgent(Agent):

    def __init__(self, n_ahead:int=1):
        super().__init__("LookaheadAgent")

    def get_best_move(self, game: Game) -> int:
        for move in game.get_valid_moves():
            game.make_move(move)
            if game.over():
                game.undo_move()
                return move
            game.undo_move()

        return int(random.choice(game.get_valid_moves()))