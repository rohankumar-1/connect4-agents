"""
A representation of the connect-four board, built for neural network input (torch)
"""
from typing import Union
import torch
import numpy as np
import hashlib

def get_rep(x: Union[int,float]) -> str: return {-1.0: 'X', 1.0: 'O'}.get(x, ' ')
def hash_tensor(t: torch.Tensor) -> bytes: return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).digest()
    
WIDTH, HEIGHT = 7, 6

class Game:

    def __init__(self, turn:Union[int,float]=1):
        self.turn: int = turn
        self.sep: str = '\n---' + ('+---' * (WIDTH - 2)) + '+---\n'
        self.board = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        self.heights = np.zeros(WIDTH, dtype=int) # for faster lookup
        self.move_stack = []
        self.num_moves = 0
        self.win_score = 0

    def get_state_tensor(self) -> torch.Tensor:
        """ representation: first layer is current player's pieces, second layer is opponents, third layer is turn """
        rep = torch.zeros((2, HEIGHT, WIDTH), dtype=torch.float32)
        rep[0] = torch.from_numpy(self.board == self.turn)
        rep[1] = torch.from_numpy(self.board == -self.turn)
        return rep

    def __repr__(self) -> str: return f"\nTurn: {get_rep(x=self.turn)}\n\n" + self.sep.join(["|".join([f" {get_rep(self.board[i, j].item())} " for j in range(WIDTH)]) for i in range(HEIGHT)])
    def get_hash(self) -> bytes: return hash_tensor(self.get_state_tensor().squeeze())
    def full(self) -> bool: return (self.num_moves == HEIGHT * WIDTH)

    def over(self) -> bool:
        """ returns true if 4 in a row or board is full. Only checks last move """
        if self.num_moves < 7: 
            return False
        r, c = self.move_stack[-1]
        p = -self.turn # player who just moved
        
        # from only last move, check in each direction for 4 in a row
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for side in [1, -1]:
                nr, nc = r + dr*side, c + dc*side
                while 0 <= nr < HEIGHT and 0 <= nc < WIDTH and self.board[nr, nc] == p:
                    count += 1
                    nr += dr*side
                    nc += dc*side
            if count >= 4:
                self.win_score = p
                return True
        
        return self.full() # only full if we have played 54 moves

    def score(self) -> int: return self.win_score
    def get_valid_moves(self) -> np.ndarray: return np.where(self.heights < HEIGHT)[0]
    def get_invalid_moves(self) -> np.ndarray: return np.where(self.heights >= HEIGHT)[0]
    def get_valid_moves_mask(self) -> torch.Tensor: return torch.from_numpy(self.heights >= HEIGHT)

    def make_move(self, col: int) -> None:
        row = (HEIGHT - 1) - self.heights[col]
        self.board[row, col] = self.turn
        self.heights[col] += 1 # Increment height
        self.move_stack.append((row, col))
        self.turn *= -1
        self.num_moves += 1

    def undo_move(self) -> None:
        row, col = self.move_stack.pop()
        self.board[row, col] = 0
        self.heights[col] -= 1 # Decrement height
        self.turn *= -1
        self.num_moves -= 1
        self.win_score = 0

    def reset(self):
        self.board = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        self.heights = np.zeros(WIDTH, dtype=int) # for faster lookup
        self.move_stack = []
        self.num_moves = 0
        self.win_score = 0
        self.turn = 1