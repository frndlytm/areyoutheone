from numpy.typing import NDArray

Player = int
Match = tuple[Player, Player]
Observation = tuple[NDArray, float]
Info = dict | None

Step = tuple[Observation, float, bool, bool, Info]
Shape = tuple[int, ...]
