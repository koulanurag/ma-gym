import logging
import numpy as np
from .predator_prey import PredatorPrey

logger = logging.getLogger(__name__)


class PredatorPreyS(PredatorPrey):
    """
    A simple environment to cross over the path of the other agent  (collaborative)

    Observation Space : Discrete
    Action Space : Discrete
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self._prey_move_probs = [0, 0, 0, 0, 1]
