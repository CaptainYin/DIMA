from enum import Enum


class Env(str, Enum):
    FLATLAND = "flatland"
    STARCRAFT = "starcraft"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"
    BIDEXHANDS = "bidexhands"
    SMACv2 = "SMACv2"

RANDOM_SEED = 23
