from agents import Agent, AlphaZeroAgent, RandomAgent, LookaheadAgent, AlphaBetaAgent
from tqdm import trange
import random
from state import Game
from argparse import ArgumentParser

parser = ArgumentParser("Run two agents against eachother")
parser.add_argument("--games", type=int, default=40, help="Number of games to play")
parser.add_argument("--bot1", type=str, default="AlphaZero", help="First bot to play")
parser.add_argument("--bot2", type=str, default="Random", help="Second bot to play")

def run_arena(bot1: Agent, bot2: Agent, n_games=40):
    # Track bot1 vs bot2 outcomes
    results = {
        "bot1": 0,
        "bot2": 0,
        "draw": 0
    }

    total_moves = 0

    for _ in trange(n_games, desc="Arena Battle"):
        game = Game()

        # Randomly assign sides: Player 1 vs Player -1
        p1_bot, pn1_bot = random.choice([(bot1, bot2), (bot2, bot1)])
        bots = {1: p1_bot, -1: pn1_bot}

        # Play the full game
        while not game.over():
            side = game.turn
            move = bots[side].get_best_move(game)
            game.make_move(move)

        # Game outcome from perspective of Player 1: -1, 0, 1
        outcome = game.score()
        total_moves += game.num_moves

        # Determine which bot the outcome corresponds to
        if outcome == 1:       # Player 1 win
            winner = bots[1]
        elif outcome == -1:    # Player -1 win
            winner = bots[-1]
        else:                  # Draw
            results["draw"] += 1
            continue

        # Increment appropriate winner counter
        if winner is bot1:
            results["bot1"] += 1
        elif winner is bot2:
            results["bot2"] += 1
        else:
            raise RuntimeError("Winner is neither bot1 nor bot2 — check logic.")

    return results, total_moves



if __name__ == "__main__":
    args = parser.parse_args()

    # validate agent choices
    if args.bot1 not in ["AlphaZero", "Random", "Lookahead", "AlphaBeta"]:
        raise ValueError(f"Invalid bot choice: {args.bot1}")
    if args.bot2 not in ["AlphaZero", "Random", "Lookahead", "AlphaBeta"]:
        raise ValueError(f"Invalid bot choice: {args.bot2}")

    # create agents based on choices
    if args.bot1 == "AlphaZero":
        bot1 = AlphaZeroAgent(model_path="models/weights/iter002.safetensors", MCTS=600, random_select=False)
    elif args.bot1 == "Random":
        bot1 = RandomAgent()
    elif args.bot1 == "Lookahead":
        bot1 = LookaheadAgent(n_ahead=1)
    elif args.bot1 == "AlphaBeta":
        bot1 = AlphaBetaAgent(depth=10)
    else:
        raise ValueError(f"Invalid bot choice: {args.bot1}")
        
    if args.bot2 == "AlphaZero":
        bot2 = AlphaZeroAgent(model_path="models/weights/iter002.safetensors", MCTS=600, random_select=False)
    elif args.bot2 == "Random":
        bot2 = RandomAgent()
    elif args.bot2 == "Lookahead":
        bot2 = LookaheadAgent(n_ahead=1)
    elif args.bot2 == "AlphaBeta":
        bot2 = AlphaBetaAgent(depth=10)
    else:
        raise ValueError(f"Invalid bot choice: {args.bot2}")

    # run arena
    print(f"Running arena between Bot 1: {args.bot1} and Bot 2: {args.bot2} for {args.games} games")
    results, total_moves = run_arena(bot1, bot2, n_games=args.games)

    print(f"Average Moves: {total_moves / args.games:.2f}")
    print(f"Bot 1 Overall Win Rate: {results['bot1'] / args.games * 100:.2f}%")
    print(f"Draw Rate: {results['draw'] / args.games * 100:.2f}%")
    print(f"Bot 2 Overall Win Rate: {results['bot2'] / args.games * 100:.2f}%")
    print("-" * 30)
    