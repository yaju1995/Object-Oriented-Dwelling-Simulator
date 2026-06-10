from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import numpy as np


@dataclass
class StrategyConfig:
    """Configuration class for Strategy parameters."""

    name: str  # Strategy name

    @classmethod
    def default_config(cls):
        """Return the default configuration for the Strategy."""
        return StrategyConfig(name="default")

    @classmethod
    def get_config_dict(cls, config):
        """Return the configuration as a dictionary"""
        return config.__dict__


class Strategy(ABC):
    """Abstract base class for Strategy implementations."""

    _strategy_registry = {}  # Class variable to store strategy implementations

    def __init__(self, config: StrategyConfig):
        """Initialize base Strategy parameters."""
        # Core parameters
        self.name = config.name

        # Configuration
        self._set_config(config)

    def _set_config(self, config: StrategyConfig):
        """Set configuration parameters."""
        for key, value in vars(config).items():
            setattr(self, key, value)

    @abstractmethod
    def reset(self):
        """Reset strategy state."""

    @abstractmethod
    def act(self, context_vector: np.ndarray, state: Optional[Dict[str, Any]]) -> int:
        """Select an action given the current state."""

    @abstractmethod
    def update(self, observed_context: np.array, reward: float):
        """Update strategy state given the observed transition."""

    @classmethod
    def register(cls, name: str, strategy_class: Type["Strategy"]):

        """Register a new strategy implementation."""
        cls._strategy_registry[name] = strategy_class
        # print(f"Strategy Register: {cls._strategy_registry[name]}")

    @classmethod
    def create(cls, config: StrategyConfig) -> "Strategy":
        """Factory method to create the appropriate strategy."""
        # print(cls._strategy_registry)
        if config.name not in cls._strategy_registry:
            raise ValueError(f"Unknown strategy type: {config.name}")
        return cls._strategy_registry[config.name](config)
