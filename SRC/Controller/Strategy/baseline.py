from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from SRC.Controller.Strategy.strategy import Strategy, StrategyConfig


@dataclass
class BaselineConfig(StrategyConfig):
    name: str = "GREEDY"

    @classmethod
    def default_config(cls):
        '''Return the default configuration for the Baseline strategy'''
        return BaselineConfig(name="GREEDY")


class Baseline(Strategy):
    """Baseline Charging strategy implementation"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def reset(self):
        """Reset strategy state"""

    def act(self, context_vector: np.ndarray, state: Optional[Dict[str, Any]]) -> int:
        # Based on the cost evaluate charging and discharging
        pass

    def update(self, observed_context: np.array, reward: float):
        """Update straeggy context_vector given the observed transition"""


Strategy.register("Baseline", Baseline)
