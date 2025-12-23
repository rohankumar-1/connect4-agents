from state import Game, get_rep
from zero import AlphaZero
from utils import SimpleBot, LookaheadBot

if __name__=="__main__":

    # bot = AlphaZero()
    # bot = SimpleBot()
    # bot1 = SimpleBot()
    bot1 = LookaheadBot()
    bot2 = LookaheadBot()
    N = 1000

    wins = {-1: 0, 1: 0, 0: 0}
    avg_moves = 0.0
    for _ in range(N):
        game = Game()
        while not game.over():
            move = bot1.get_best_move(game)
            game.make_move(move)
            if game.over():
                continue
            move = bot2.get_best_move(game)
            game.make_move(move)

        wins[game.score()] += 1
        avg_moves += game.num_moves

    print(f"Average # of moves: {avg_moves / N}")
    print("Final win percentage:")
    print(get_rep(-1), " win: ", wins[-1] / N)
    print("Draw: ", wins[0] / N)
    print(get_rep(1), " win: ", wins[1] / N)