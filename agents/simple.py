from state import Game
import random
from .agent import Agent

class RandomAgent(Agent):

    def __init__(self):
        super().__init__(name="RandomAgent")

    def get_best_move(self, game: Game) -> int:
        return int(random.choice(seq=game.get_valid_moves()))
