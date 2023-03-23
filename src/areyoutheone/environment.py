"""
# Simple "Are You The One?"

areyoutheone is a match-up dating game.

The players, $G = (A \Intersect A\prime, E)$, are split into two
even teams of size $m = |A| = |A\prime|$ seeking their in a
perfect matching $E$. They have $w$ weeks to find the perfect
match $E$ to win up to four units of reward money, $R$.

Every iteration,

* the choosers and the chosen switch roles.
* the game chooses a playing order for the choosers.
* the choosers choose matches in playing order from the set of
chooseable players.

This decision process results in a sample matching $\hat{E}$,
after which point, the players receive infomation about the
sample matching in the form of $m$ beams.

* Each beam symbolizes one correct answer, in the matching
chosen by the palyers.
* Zero beams is a "blackout", resulting in a reward penalty
of $-0.25 R$ to the total reward to the players.
* $m$ beams is a perfect match, symbolizing the game is over,
in which case, the players get the remaining reward.

!!! Note:
  The game also ends and the players get the remaing reward
  if the remaining reward hits 0.

Given those dynamics, we can design a belief network to solve
this problem and use an agent to play the game for us.
"""
import random
from functools import singledispatchmethod
from dataclasses import dataclass
from typing import *

import numpy as np

from areyoutheone.types import Info, Match, Player


@dataclass
class MatchUp:
    game: "AreYouTheOne"
    choosers: Sequence[Player]
    chosen: Sequence[Player]
    fluid: bool = False

    def __post_init__(self):
        if (n_choosers := len(self.choosers)) != (n_chosen := len(self.chosen)):
            raise ValueError(
                "Complete match-up has equal sequence lengths: "
                f"(Choosers, Chosen) = ({n_choosers}, {n_chosen})."
            )

    def __len__(self):
        return len(self.choosers)

    def __iter__(self):
        return zip(self.choosers, self.chosen)

    @singledispatchmethod
    def __contains__(self, x: Any):
        raise ValueError(f"{x} is not a Player or a Match.")
        
    @__contains__.register
    def __contains_player__(self, player: Player):
        for chooser in self.choosers:
            if player == chooser:
                return True

        for chosen in self.chosen:
            if player == chosen:
                return True

        return False

    @__contains__.register
    def __contains_match__(self, match: Match):
        # Do a linear scan for the match or its swap in the match-up
        for chooser, chosen in self:
            if match == (chooser, chosen) or match == (chosen, chooser):
                return True

        return False

    def __getitem__(self, player: Player) -> Player:
        return self.get_match(player)

    @property
    def players(self) -> set[Player]:
        return set(self.choosers) | set(self.chosen)

    def add(self, chooser: Player, chosen: Player):
        # Cannot add a match to the match-up if it comtains a
        # player who already has a match
        if self.match(chooser) is not None:
            raise ValueError(...)
        if self.match(chosen) is not None:
            raise ValueError(...)

        self.choosers.append(chooser)
        self.chosen.append(chosen)

    def match(self, player: Player) -> Player:
        for chooser, chosen in self:
            if player == chooser:
                return chosen
            elif player == chosen:
                return chooser
            else:
                continue
        return None

    def shuffle(self) -> "MatchUp":
        # Return a new match-up with shuffled pairs
        return MatchUp(
            game=self.game,
            choosers=random.shuffle(self.choosers),
            chosen=random.shuffle(self.chosen),
            fluid=self.fluid,
        )

    def nodes(self):
        return (self.choosers, self.chosen)

    def edges(self):
        return list(zip(*self.nodes()))


class AreYouTheOne:
    def __init__(
        self,
        matching: set[Match],
        guesses: int | None = None,
        prize: float = 1_000_000,
        fluid: bool = False,
    ):
        self.n_pairs = len(matching)
        self.matching = matching
        self.fluid = fluid
        self.guesses = guesses or self.n_pairs
        self.total_prize = prize
        self.remaining_prize = prize
        self.possible_matches = np.ones((self.n_pairs, self.n_pairs))

    def truth(self, match: Match) -> bool:
        return match in self.matching

    def beams(self, matchup: MatchUp) -> Sequence[bool]:
        return np.array([(match in self.matching) for match in matchup], dtype=int)

    def options(self, player: Player) -> set[Player]:
        # Season 8: Secually Fluid Season
        # Basically, you can't match wkth yourself, but every other
        # player is fair game.
        if self.fluid:
            return self.choosers | self.chosen / {player}
        else:
            return self.choosers if player in self.chosen else self.chosen

    def choices(self, player: Player, matchup: MatchUp) -> set[Player]:
        # fmt:off
        return (
            self.options(player)  # Take the player's options
            / set(matchup.players) # Remove already  matched players
            | matchup.match(player)  # Add back the player's current match
        )
        # fmt:on

    def is_blackout(self, matchup: MatchUp, info: Info = None) -> bool:
        # TODO: Truth Booth
        #     ISSUE: issues/1
        #     AUTHOR: frndlytm
        #     DESCRIPTION:
        #         Enabling the Truth booth is going to require encoding the agent's
        #         knowledge about perfect matches from the truth booth into the info.
        #         A blackout means no new beams above the knowledge of perfect matches.
        #
        n_beams = self.beams(matchup).sum()
        return bool(n_beams)

    def is_perfect(self, matchup: MatchUp, _: Info = None) -> bool:
        n_beams = self.beams(matchup).sum()
        return n_beams == self.n_pairs

    def step(self, A: MatchUp, info: Info = None) -> float:
        info = info or {}
        self.guesses -= 1

        # By default, just gusessing doesn't win a prize, and the game ends when
        # there are no more guesses or no more prize money remaining
        reward, success, truncated = 0, False, not (self.guesses and self.remaining_prize)

        # "Blackouts" result in a reduction to the prize money
        if self.is_blackout(A, info):
            self.remaining_prize -= 0.25 * self.total_prize

        # A "Perfect Match" means the agent wins the game. The beams mean we
        # reduce the possible matches to a permutation matrix, and 
        elif self.is_perfect(A, info):
            reward, success, truncated = self.remaining_prize, True, False

            self.possible_matches = 0
            for chooser, chosen in A:
                self.possible_matches[(chooser, chosen)] = 1
                self.possible_matches[(chosen, chooser)] = 1

        # An observation looks like a boolean matrix containing the possible
        # choices for each player in the game
        return self.possible_matches, reward, success, truncated, info
