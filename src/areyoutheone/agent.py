from typing import Sequence

import numpy as np

from networkx.algorithms import bipartite
from numpy.typing import NDArray

from areyoutheone import types
from areyoutheone.environment import AreYouTheOne, Observation, MatchUp

Experience = tuple[Observation, MatchUp, Observation]


class AreYouTheOneAgent:
    """
    An AreYouTheOneAgent's objective is to find the correct perfect matching
    in the set of all possible perfect matchings crossing between the two sets
    of players in a game of AreYouTheOne.

    We pursue this heuristically by guessing a maximally stable marriage at each
    turn using the possible_matches observation provided by the AreYouTheOne
    environment.
    """
    def __init__(self, environment: AreYouTheOne):
        self.environment = environment
        self.positive_beliefs = np.ones_like(environment.shape)
        self.negative_beliefs = np.ones_like(environment.shape)
        self.memory = []

    def get_beliefs(self, experiences: Sequence[Experience]) -> NDArray:
        ...

    def get_match_up(self, observation: Observation) -> MatchUp:
        ...
