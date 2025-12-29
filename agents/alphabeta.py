"""
Alpha-Beta (via Minimax) pruning implementation for Connect 4

Based on pseudocode here; except with negamax for simplicity:
https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
"""

from state import Game
from .agent import Agent
import numpy as np

class AlphaBetaAgent(Agent):

    def __init__(self, depth):
        super().__init__("AlphaBetaAgent")
        self.depth = depth

    def get_best_move(self, game: Game) -> int:
        best_move, best_value = -1, -np.inf
        for move in game.get_valid_moves():
            game.make_move(move)
            # value = -self.alphabeta_value_minimax(game, self.depth, alpha=-np.inf, beta=np.inf, maximizing=False)
            value = -self.negamax(game, self.depth, -np.inf, np.inf)
            if value > best_value:
                best_move = move
                best_value = value
            game.undo_move()
        return best_move


    def alphabeta_value_minimax(self, game: Game, depth, alpha, beta, maximizing: bool) -> float:
        if game.over() or depth == 0:
            return self._evaluate_board(game)

        if maximizing:
            value = -np.inf
            for move in game.get_valid_moves():
                game.make_move(move)
                value = max(value, self.alphabeta_value_minimax(game, depth-1, alpha, beta, maximizing=not maximizing))
                game.undo_move()
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value
        else:
            value = np.inf
            for move in game.get_valid_moves():
                game.make_move(move)
                value = min(value, self.alphabeta_value_minimax(game, depth-1, alpha, beta, maximizing=not maximizing))
                game.undo_move()
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value

    def negamax(self, game: Game, depth, alpha, beta) -> float:
        if game.over() or depth == 0:
            return self._evaluate_board(game)

        value = -np.inf
        for move in game.get_valid_moves():
            game.make_move(move)
            value = max(value, -self.negamax(game, depth - 1, -beta, -alpha))
            game.undo_move()
            
            if value >= beta:
                return value  # Beta cutoff
            alpha = max(alpha, value)
        
        return value


    #########################################################################
    # 
    #   Below we define the heuristics:
    #   1. Center pieces are generally good
    #   2. 3 in a rows are great if there is extra space
    #   3. 2 in a rows are good if there are 2 extra spaces
    #   4. If the opposite has 2/3 in a row, effect is negative
    #
    ######################################################################### 

    def _evaluate_board(self, game: Game) -> float:
        if game.over(): 
            return 0.0 if game.full() else -10000.0
        
        score = 0.0

        windows = get_windows(game.board)
        num_players = (windows == game.turn).sum(axis=1)
        num_zeros = (windows == 0).sum(axis=1)
        num_opponents = (windows == -game.turn).sum(axis=1)

        # encourage unblocked 2/3 in a single window
        score += 5.0    * ((num_players == 2) & (num_zeros == 2)).sum()
        score += 50.0   * ((num_players == 3) & (num_zeros == 1)).sum()

        # punish opponent having unblocked sequence a single window
        score -= 5.0    * ((num_opponents == 2) & (num_zeros == 2)).sum()
        score -= 50.0   * ((num_opponents == 3) & (num_zeros == 1)).sum()

        # encourage middle column usage
        score += 5.0    * (game.board[:, 3] == game.turn).sum()
        score -= 5.0    * (game.board[:, 3] == -game.turn).sum()

        return score


def get_windows(board: np.ndarray):
    rows, cols = board.shape
    windows = []
    
    # Horizontal windows
    if cols >= 4:
        horizontal = np.lib.stride_tricks.sliding_window_view(board, (1, 4))
        windows.append(horizontal.reshape(-1, 4))
    
    # Vertical windows
    if rows >= 4:
        vertical = np.lib.stride_tricks.sliding_window_view(board, (4, 1))
        windows.append(vertical.reshape(-1, 4))
    
    # Diagonal windows (top-left to bottom-right)
    if rows >= 4 and cols >= 4:
        diag1 = []
        for offset in range(-(rows - 4), cols - 3):
            diag = np.diagonal(board, offset=offset)
            if len(diag) >= 4:
                diag1.append(np.lib.stride_tricks.sliding_window_view(diag, 4))
        if diag1:
            windows.append(np.vstack(diag1))
    
    # Diagonal windows (bottom-left to top-right)
    if rows >= 4 and cols >= 4:
        diag2 = []
        flipped = np.flipud(board)
        for offset in range(-(rows - 4), cols - 3):
            diag = np.diagonal(flipped, offset=offset)
            if len(diag) >= 4:
                diag2.append(np.lib.stride_tricks.sliding_window_view(diag, 4))
        if diag2:
            windows.append(np.vstack(diag2))
    
    return np.vstack(windows) if windows else np.array([]).reshape(0, 4)

