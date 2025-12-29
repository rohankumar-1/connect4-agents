
TODO:
- q vs z training target
- build test dataset 
- add probabilities, state value prediction, and step button to GUI
- implement model evaluation: 100 games vs itself, if win, replace



AlphaZero works like this:

We start at some state (board+turn). The goal is to find the best move. We simulate 800 moves, then pick the move that was visited the most (since we use PUCT as a selection heuristic, this guides us to the best option). 

In each simulation from the current state, we perform the following recursive algorithm.
1. Check if the current state has been visited before. 
2a. If it is has, use PUCT to choose the next move and recurse to this
2b. If it has not:
    i. check for win condition. If win, set value to actual score, return. Prior probabilities dont matter at this state, so set to None.
    ii. use model to predict prior probabilities and value. Then, recurse back up the stack of moves to the root state, updating visit count by 1 and adding the predicted value to the running sum of predicted values. 


Once this process completes, we take the best move, and repeat until win condition materializes. 

Other considerations:
1) at each alternating recursive call, flip the sign of the value (since a win for opponent is loss for current player)
2) make sure to mask out PUCT for invalid moves (we do not want to explore these AT ALL)
3) create a new dictionary of visit and value counts at each potential move




## RESULTS

Random Bot vs Random Bot:
Average # of moves: 21.913
Final win percentage:
X  win:  0.459
Draw:  0.004
O  win:  0.537

Lookahead Bot vs Random Bot:
Average # of moves: 17.514
Final win percentage:
X  win:  0.708 (lookahead)
Draw:  0.0
O  win:  0.292 (random)

Lookahead Bot vs Lookahead Bot
Average # of moves: 14.44
Final win percentage:
X  win:  0.408
Draw:  0.0
O  win:  0.592

Untrained AlphaZero (random model, MCTS=10) vs Random Bot
Average # of moves: 27.13
Final win percentage:
X  win:  0.87
Draw:  0.06
O  win:  0.07

Untrained AlphaZero (random model, MCTS=10) vs Lookahead Bot
Average # of moves: 16.89
Final win percentage:
X  win:  0.97
Draw:  0.02
O  win:  0.01

Untrained AlphaZero vs Untrained AlphaZero
Average # of moves: 23.09
Final win percentage:
X  win:  0.94
Draw:  0.03
O  win:  0.03

