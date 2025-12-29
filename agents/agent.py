from abc import ABC, abstractmethod
from state import Game



class Agent(ABC):

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def get_best_move(self, game: Game) -> int:
        """
        Given a Game state, return the best move as an integer.
        Must be implemented by subclasses (e.g., AlphaZero, Minimax).
        """
        raise NotImplementedError