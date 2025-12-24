from state import Game, get_rep
from zero import AlphaZero
from utils import SimpleBot, LookaheadBot
from tqdm import trange
import torch


if __name__=="__main__":

    bot = AlphaZero(noise=0.3)
    # bot = SimpleBot()
    # bot1 = SimpleBot()
    # bot2 = LookaheadBot()
    # bot2 = AlphaZero(noise=1.0)
    # bot2 = SimpleBot()
    N = 10

    wins = {-1: 0, 1: 0, 0: 0}
    avg_moves = 0.0
    for _ in trange(N):
        game = Game()
        while not game.over():
            move = bot.get_best_move(game)
            game.make_move(move)

        wins[game.score()] += 1
        avg_moves += game.num_moves

    print(f"Average # of moves: {avg_moves / N}")
    print("Final win percentage:")
    print(get_rep(-1), " win: ", wins[-1] / N)
    print("Draw: ", wins[0] / N)
    print(get_rep(1), " win: ", wins[1] / N)