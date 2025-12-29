from state import Game
from agents import AlphaBetaAgent
from agents.alphabeta import get_windows


if __name__ == "__main__":
    g = Game()
    bot = AlphaBetaAgent(depth=3)

    # while not g.over():
    #     print(g)
    #     m = bot.get_best_move(g)
    #     g.make_move(m)

    # "FINISHED:"
    # print(g)

    g.make_move(1)
    g.make_move(3)
    g.make_move(2)
    g.make_move(3)
    print(g)
    print(get_windows(g.board))
    print(bot._evaluate_board(g))