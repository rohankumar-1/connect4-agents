"""
With the most recent model, how far off from the ground-truth best moves are we?
"""

import math
import pandas as pd
import torch
from state import Game
from model import PolicyValueNetwork
from ucimlrepo import fetch_ucirepo
from argparse import ArgumentParser
from tqdm import trange
import numpy as np
from zero import AlphaZero

# fetch dataset 
connect_4 = fetch_ucirepo(id=26) 
X: pd.DataFrame = connect_4.data.features 
y: pd.Series = connect_4.data.targets 
parser = ArgumentParser("evaluation")
parser.add_argument("--model_id", "-mid")

def get_reverse_rep(x:str): return {'x':1., 'b':0., 'o':-1.}.get(x)
def get_outcome_val(x:str): return {'win': 1, 'loss': -1, 'draw': 0}.get(x)

def build_game_from_df(row: pd.Series, outcome: str):
    """ take in a UCI Connect 4 game sample, returns game ready to evaluate """
    board = np.zeros((6,7))
    i, j = 0, 0
    for e in row:
        board[5-i][j] = get_reverse_rep(e)
        i += 1
        if i == 6:
            i = 0
            j += 1
    
    g = Game(turn=1)
    g.board = board # straight overwrite board
    # print(outcome)
    return g, get_outcome_val(outcome)

def evaluate(net, X, y):
    correct = 0
    total = len(y)
    for i in trange(len(y)):
        x_samp, y_samp = X.iloc[i, :], y.iloc[i, 0]
        game, outcome = build_game_from_df(x_samp, y_samp)
        _, v = net.predict(game.get_state_tensor())

        # convert value to categorical
        pred_class = 1 if v > 0.1 else (-1 if v < -0.1 else 0)
        correct += int(pred_class == outcome)

    return correct / (total + 1e-5)


if __name__=="__main__":
    args = parser.parse_args()

    # load in model
    net = PolicyValueNetwork()
    net._load_checkpoint(f"models/start{int(args.model_id):03}.safetensors")

    correct = 0
    total = len(y)
    for i in trange(20):
        x_samp, y_samp = X.iloc[i, :], y.iloc[i, 0]
        game, outcome = build_game_from_df(x_samp, y_samp)
        
        _, v = net.predict(game.get_state_tensor())

        # convert value to categorical
        # print(round(v.item()))
        pred_class = 1 if v > 0.1 else (-1 if v < -0.1 else 0)
        correct += int(pred_class == outcome)

    print(f"For model iteration {args.model_id}, accuracy of value head is {correct / total}")
