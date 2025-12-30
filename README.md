# Connect4 Agents

In this repo, I reimplement algorithms to play connect-4 using basic reinforcement learning. The main algorithms are AlphaZero, from [this paper](https://arxiv.org/pdf/1712.01815), and alpha-beta pruning (negamax implementation). 


### AlphaZero

I only train up to 3 iterations, so there is lots of room for improvement, but it can clearly beat random or lookahead agents (see Arena Results). The underlying Policy-Value prediction model is a convolutional network with a head for each task, like in the original paper. Self-play is generally slow, as it is difficult to paralellize the MCTS iterations, so this remains as the bottleneck in training.

### Alpha-Beta Pruning

For alpha-beta pruning, I decided to use a heuristic function that rewards 2 or 3 in a rows with no blocking pieces, as well as the number of pieces in the center column. At depth=5, this approach is already very very good (can beat AlphaZero almost always). There is still room for better heuristics.

## Directory Structure

- `agents/` - Agent implementations (AlphaZero, AlphaBeta, Random, Lookahead)
- `models/` - Neural network models and weights
- `data/` - Training data
- `app/` - Web GUI
- `arena.py` - Arena script for agent vs agent battles
- `train_alphazero.py` - AlphaZero training script
- `state.py` - Game state implementation

## App

There is a super simple GUI to go along with this project, so you can easily play against AlphaZero. See the file "app/main.py" to adjust which weights the model uses upon startup. 

To start the server:
```bash
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` in your browser. 

## Script Usage

There are two main scripts. "arena.py" runs two agents against eachother for a set number of games.

### Arena

Example:
```bash
python arena.py --bot1 AlphaZero --bot2 Random --games 40
```

The agents must be one of the following: `AlphaZero`, `AlphaBeta`, `Random`, `Lookahead`. If desired, add the `--verbose` flag to see how the game progresses (recommendation is to only do this when games=1).

### Training

Example:
```bash
python train_alphazero.py --start 1 --iterations 10 --MCTS 400 --verbose
```

Start is the iteration you wish to begin training from. See `train_alphazero.py` for more info.

## Arena Results

Here, we test each agent against each other agent. 

>Hyperparameters: 
> - AlphaZero has MCTS=600
> - AlphaBeta has depth=2 (to keep it competitive)

| | Random | Lookahead | AlphaZero | AlphaBeta |
|---|---|---|---|---|
| **Random** | - | 0.26 | 0.0 | 0.0 |
| **Lookahead** | 0.74 | - | 0.02 | 0.0 |
| **AlphaZero** | 1.0 | 0.98 | - | 0.0 |
| **AlphaBeta** | 1.0 | 1.0 | 1.0 | - |

