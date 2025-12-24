"""
A representation of the connect-four board, built for neural network input (torch)
"""
from typing import Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import hashlib

def get_rep(x: Union[int,float]) -> str: return {-1.0: 'X', 1.0: 'O'}.get(x, ' ')

""" 
hash the state representation, returning byte string. this implementation is not shape/dtype sensitive, 
but we use the same shape/dtype for every state rep in this project
"""
def hash_tensor(t: torch.Tensor) -> bytes: return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).digest()
    

class Game:

    def __init__(self, H:int=6, W:int=7, turn:Union[int,float]=1):
        assert W > 4
        assert H > 4
        self.turn: int = turn
        self.width: int = W
        self.height:int = H
        self.board: torch.Tensor = torch.zeros(size=(H, W))
        self.win_score: int = 0 # default is a draw
        self.sep: str = '\n---' + ('+---' * (self.width - 2)) + '+---\n'
        self.conv_hori = torch.tensor([[1., 1., 1., 1.]]).view(1,1,1,4).repeat(1,1,1,1)
        self.conv_vert = torch.tensor([[1.], [1.], [1.], [1.]]).view(1,4,1).repeat(1,1,1,1)
        self.conv_diag = torch.eye(4).view(1,1,4,4).repeat(1,1,1,1)
        self.num_moves = 0

        # stack to hold all moves done already
        self.move_stack = []


    def __repr__(self) -> str:
        """ build a string that pretty prints the board"""
        return f"\nTurn: {get_rep(x=self.turn)}\n\n" + self.sep.join(["|".join([f" {get_rep(self.board[i, j].item())} " for j in range(self.width)]) for i in range(self.height)])


    def get_state_tensor(self) -> torch.Tensor:
        """ representation: first layer is current player's pieces, second layer is opponents, third layer is turn """
        rep:torch.Tensor = torch.zeros(size=(2, self.height, self.width))
        rep[0, :, :] = self.board == self.turn # first layer
        rep[1, :, :] = self.board == -self.turn
        return rep.unsqueeze(dim=0)

    def get_hash(self) -> bytes: return hash_tensor(self.get_state_tensor().squeeze())

    def none_empty(self) -> bool:
        return bool(~(self.board == 0.).any())

    def over(self) -> bool:
        """ returns true if 4 in a row via convolutions, or board is full.  """
        v_check = F.conv2d(input=self.board.view(1,1,self.height, self.width), weight=self.conv_vert)
        h_check = F.conv2d(input=self.board.view(1,1,self.height, self.width), weight=self.conv_hori)
        d_check = F.conv2d(input=self.board.view(1,1,self.height, self.width), weight=self.conv_diag)

        win = bool((
            (torch.abs(v_check)==4.).any() | 
            (torch.abs(h_check)==4.).any() | 
            (torch.abs(d_check)==4.).any()
        ).item())

        if win: # assume player just placed piece, then we check win
            self.win_score:int= -self.turn

        return win | self.none_empty()

    def score(self) -> int:
        return self.win_score

    def get_valid_moves(self) -> np.ndarray:
        """ return idx of all columns that are not full (turn independent)"""
        return np.arange(self.width)[(self.board == 0.).any(dim=0)]

    def get_invalid_moves(self) -> np.ndarray:
        return np.arange(self.width)[~(self.board == 0.).any(dim=0)]

    def get_valid_moves_mask(self) -> torch.Tensor:
        mask = torch.ones((7,), dtype=torch.int32) # mask probability of illegal moves to 0
        mask[self.get_valid_moves()] = 0
        return mask

    def make_move(self, col: int) -> None:
        """ drop current token into slot, update turn """
        row = max(torch.argwhere(self.board[:, col] == 0.))
        self.board[row, col] = self.turn
        self.move_stack.append((row, col))
        self.turn *= -1
        self.num_moves += 1

    def undo_move(self) -> None:
        r,c = self.move_stack.pop()
        self.board[r, c] = 0.
        self.win_score = 0 # reset in case we had just played last piece
        self.num_moves -= 1
        self.turn *= -1

    @staticmethod
    def build_game(tens: torch.Tensor):
        assert tens.shape[0] == 2
        H, W = tens[0].shape
        g = Game(H, W, turn=1)
        g.board = torch.sum(tens, dim=0).squeeze()
        return g



if __name__=="__main__":
    g = Game(6, 7, 1)
    g.make_move(1)
    g.make_move(2)
    g.make_move(1)
    g.make_move(2)
    g.make_move(1)
    g.make_move(2)
    print(g.over(), g.score())
    g.make_move(1)
    print(g)
    print("\nValid moves: ", g.get_valid_moves())
    print(g.over(), g.score())

