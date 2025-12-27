"""
implementing the training / iterative improvement algorithm
"""
import math
import torch
from torch import Tensor
from model import PolicyValueNetwork
from state import Game

class AlphaZero:

    def __init__(self, noise:float=0.0, MCTS=300, C_PUCT=1.0, model_pth=None, train=False, random_select=True):
        self.dirichlet_dist = torch.distributions.Dirichlet(concentration=(torch.ones((7,))*noise))
        self.model = PolicyValueNetwork(num_res_blocks=4, path=model_pth)
        self.model.eval()
        self.MCTS: int = MCTS
        self.C_PUCT: float = C_PUCT
        self.model: PolicyValueNetwork = torch.compile(self.model)
        self.train: bool = train
        if self.train:
            self.data: list = []
        self.random_select: bool = random_select

    def _puct(self, s):
        """ predicted upper confidence bound applied to trees """
        visits = self.visits[s]
        return (self.q[s] + self.C_PUCT * self.priors[s] * math.sqrt(visits.sum())) / (1+visits)

    def _reset_dicts(self):
        self.priors: dict[bytes, Tensor]  = dict()    # probability distribution over moves from s
        self.visits: dict[bytes, Tensor]  = dict()    # times we have gotten to state s
        self.q: dict[bytes, Tensor]       = dict()    # running sum of values seen from state s

    def get_best_move(self, game: Game) -> int:
        """ simulate a bunch of model-guided MCTS, then pick the action that brings us to the most visited state """
        self._reset_dicts()

        # set initial priors from model + Dirichlet noise (at root node, we have no information yet)
        root_hash: bytes = game.get_hash()
        prior_pred, _ = self.model.predict(game.get_state_tensor())

        if self.train:
            noise: Tensor = self.dirichlet_dist.sample()
            self.priors[root_hash] = 0.75 * prior_pred + 0.25 * noise
        else:
            self.priors[root_hash] = prior_pred

        self.visits[root_hash] = torch.zeros(7, dtype=torch.float32)
        self.q[root_hash] = torch.zeros(7, dtype=torch.float32)

        # do large # of tree searches (via model-guided MCTS)
        for _ in range(self.MCTS):
            self._value(game)

        # get the move that was visited the most (if model is good, visits == strength of move)
        if self.train:
            self.data.append({"s_t": game.get_state_tensor(), "alpha_t": Tensor(self.visits[root_hash]/self.MCTS), "turn": game.turn, 'q_t': self.q[root_hash]/self.MCTS})

        if self.random_select:
            temperature = 1.0 if game.num_moves < 4 else 0.95 # 1.0 is standard; lower (e.g. 0.1) becomes like argmax
            adjusted_visits = torch.pow(self.visits[root_hash], 1/temperature)
            probs = adjusted_visits / torch.sum(adjusted_visits)
            return int(torch.multinomial(probs, 1).item())

        return int(torch.argmax(self.visits[root_hash]).item())


    def get_data(self, game_result: int, avg_qz=True):
        """ reset data for a new game: if turn is the same as the game winner, then encourage this sample, else discourage """
        res: list = self.data.copy()
        for sample in res:
            if game_result == 0:
                sample['z_t'] = 0.0
            else:
                sample["z_t"] = 1.0 if (sample["turn"] == game_result) else -1.0 
        self.data: list = []
        return res


    def _value(self, game: Game) -> float:
        if game.over():
            return -1.0 if not game.full() else 0.0

        s = game.get_hash()
        mask = torch.from_numpy(game.get_invalid_moves())
        puct_scores = self._puct(s)
        puct_scores[mask] = -torch.inf
        a: int = torch.argmax(puct_scores).numpy()
        game.make_move(a)
        s_next = game.get_hash()
        if s_next not in self.priors:
            p_logits, v = self.model.predict(game.get_state_tensor())
            self.priors[s_next] = p_logits
            self.visits[s_next] = torch.zeros(7, dtype=torch.float32)
            self.q[s_next] = torch.zeros(7, dtype=torch.float32)
        else:
            # Recursive step
            v = self._value(game)

        game.undo_move()
        self.q[s][a] = self.q[s][a] - v
        self.visits[s][a] += 1
        return -v
