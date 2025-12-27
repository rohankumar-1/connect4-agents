import random
from tqdm import trange
from pyinstrument import Profiler

from state import Game
from zero import AlphaZero


class SimpleBot:
    def get_best_move(self, game: Game):
        return random.choice(game.get_valid_moves())

class LookaheadBot:
    """ looks ahead one time """
    def get_best_move(self, game):

        for move in game.get_valid_moves():
            game.make_move(move)
            if game.over():
                game.undo_move()
                return move
            game.undo_move()

        return random.choice(game.get_valid_moves())


def run_arena(bot1, bot2, n_games=40):
    # Track wins by Player ID (-1, 0, 1)
    results = {-1: 0, 0: 0, 1: 0}
    # Track wins specifically for bot1 to see its performance
    bot1_wins = 0
    total_moves = 0

    for _ in trange(n_games, desc="Arena Battle"):
        game = Game()
        
        # Randomly assign which bot is Player 1 and which is Player -1
        # bots_map maps PlayerID -> Bot Object
        p1_bot, pn1_bot = (bot1, bot2) if random.random() > 0.5 else (bot2, bot1)
        bots = {1: p1_bot, -1: pn1_bot}

        while not game.over():
            current_turn = game.turn # Assumes your game tracks current player
            active_bot = bots[current_turn]
            
            move = active_bot.get_best_move(game)
            game.make_move(move)

        # Update stats
        outcome = game.score()
        results[outcome] += 1
        total_moves += game.num_moves
        
        # Track if bot1 specifically won (regardless of its PlayerID)
        if (outcome == 1 and bots[1] == bot1) or (outcome == -1 and bots[-1] == bot1):
            bot1_wins += 1

    # Printing Results
    print(f"\n--- Battle Results ({n_games} games) ---")
    print(f"Average Moves: {total_moves / n_games:.2f}")
    print(f"Bot 1 Overall Win Rate: {bot1_wins / n_games:.1%}")
    print(f"Draw Rate: {results[0] / n_games:.1%}")
    print("-" * 30)
    
    return results

if __name__ == "__main__":
    b1 = AlphaZero(noise=0.1, model_pth="models/iter002.safetensors")
    b2 = AlphaZero(noise=0.1, model_pth="models/best001.safetensors")
    
    run_arena(b1, b2, n_games=40)



#### PROFILING

# if __name__=="__main__":
#     bot = AlphaZero(noise=0.3)
#     game = Game()
#     profiler = Profiler(interval=1e-4)
#     profiler.start()

#     # Run a single move or a whole game
#     for _ in range(10):
#         bot.get_best_move(game)

#     profiler.stop()
#     profiler.print()
#     profiler.open_in_browser()