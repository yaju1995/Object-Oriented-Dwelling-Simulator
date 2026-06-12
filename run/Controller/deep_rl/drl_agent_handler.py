"""
Base parent class for all Deep Reinforcement Learning agents.

All DRL algorithms in the project should inherit from DRLAgent and implement
its abstract methods. The parent class only defines the common interface and
stores shared configuration values.

Example:
    class DQNAgent(DRLAgent):
        def get_action(self, state, explore=True):
            ...

        def store_transition(self, state, action, reward, next_state, termination):
            ...

        def train(self):
            ...

        def save(self, path=None):
            ...

        def load(self, path):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Literal

import numpy as np
import torch


ReturnMode = Literal["n_step", "mc", "hybrid", "sparse"]
ActionType = Literal["discrete", "continuous"]


@dataclass
class DRLAgentConfig:
    """
    Common configuration that every DRL agent should define.

    Algorithm-specific configs can inherit from this class.

    Example:
        @dataclass
        class DDPGConfig(DRLAgentConfig):
            actor_lr: float = 1e-4
            critic_lr: float = 1e-3
            tau: float = 0.005
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    name: str = "DRLAgent"
    algorithm: str = "base"

    # ------------------------------------------------------------------
    # Environment / state-action dimensions
    # ------------------------------------------------------------------
    obs_dim: int = 1
    action_dim: int = 1
    action_type: ActionType = "continuous"

    # For continuous actions
    a_min: Optional[float | np.ndarray] = -1.0
    a_max: Optional[float | np.ndarray] = 1.0

    # For discrete actions
    n_actions: Optional[int] = None
    action_map: Optional[dict] = None

    # Optional dynamic bound function:
    # bound_fn(state) -> (a_min, a_max)
    bound_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None

    # ------------------------------------------------------------------
    # Learning hyperparameters
    # ------------------------------------------------------------------
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 128
    buffer_capacity: int = 100_000

    # Used by actor-critic methods such as DDPG/TD3/SAC
    tau: float = 0.005

    # ------------------------------------------------------------------
    # Return calculation
    # ------------------------------------------------------------------
    return_mode: ReturnMode = "n_step"
    n_step: int = 1

    # ------------------------------------------------------------------
    # Network settings
    # ------------------------------------------------------------------
    hidden: Tuple[int, ...] = (64, 64)
    activation: str = "relu"

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------
    train_start: int = 1_000
    train_freq: int = 1
    target_update_freq: int = 1
    gradient_steps: int = 1

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    noise_std: float = 0.1

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0

    # ------------------------------------------------------------------
    # Saving / logging
    # ------------------------------------------------------------------
    save_dir: str = "./models"
    log_dir: str = "./logs"
    extra: Dict[str, Any] = field(default_factory=dict)


class DRLAgent(ABC):
    """
    Parent class that all DRL agents must follow.

    This class intentionally does not implement the learning algorithm.
    DQN, DDPG, TD3, SAC, PPO, etc. should inherit from this class and
    implement the required methods.
    """

    def __init__(self, config: DRLAgentConfig):
        self.config = config

        self.train_step = 0
        self.total_steps = 0

        self.device = torch.device(config.device)

        self._set_seed(config.seed)

    # ------------------------------------------------------------------
    # Required methods for all DRL agents
    # ------------------------------------------------------------------
    @abstractmethod
    def get_action(self, state: np.ndarray, explore: bool = True) -> Any:
        """
        Return an action for the current state.

        Parameters
        ----------
        state:
            Current environment/controller state.
        explore:
            If True, use exploration noise or epsilon-greedy behaviour.
            If False, return deterministic/evaluation action.
        """
        raise NotImplementedError

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        termination: bool,
    ) -> None:
        """
        Store one transition in the agent memory/replay buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self) -> Optional[Dict[str, float]]:
        """
        Update the DRL model.

        Returns
        -------
        Optional dictionary of training losses/statistics.
        Example: {"critic_loss": 0.01, "actor_loss": -0.3}
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Optional[str] = None) -> str:
        """
        Save model, optimizers, replay metadata, and configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> str:
        """
        Load saved model and optimizer state.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional shared helper methods
    # ------------------------------------------------------------------
    def get_action_bounds(self, state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return action bounds.

        If config.bound_fn is provided, the bounds can change based on state.
        Otherwise static config.a_min and config.a_max are used.
        """
        if self.config.bound_fn is not None:
            if state is None:
                raise ValueError("state is required when using dynamic bound_fn")
            a_min, a_max = self.config.bound_fn(state)
        else:
            a_min, a_max = self.config.a_min, self.config.a_max

        return (
            np.asarray(a_min, dtype=np.float32),
            np.asarray(a_max, dtype=np.float32),
        )

    @staticmethod
    def scale_action(raw_action: np.ndarray, a_min: np.ndarray, a_max: np.ndarray) -> np.ndarray:
        """
        Scale action from [-1, 1] to [a_min, a_max].

        Useful for actor networks using tanh output.
        """
        raw_action = np.asarray(raw_action, dtype=np.float32)
        return a_min + 0.5 * (raw_action + 1.0) * (a_max - a_min)

    @staticmethod
    def clip_action(action: np.ndarray, a_min: np.ndarray, a_max: np.ndarray) -> np.ndarray:
        """
        Clip continuous action to valid range.
        """
        return np.clip(action, a_min, a_max)

    def ready_to_train(self, memory_size: int) -> bool:
        """
        Common condition used before calling train().
        """
        return memory_size >= max(self.config.train_start, self.config.batch_size)

    def increment_step(self) -> None:
        """
        Update environment interaction counter.
        """
        self.total_steps += 1

    def increment_train_step(self) -> None:
        """
        Update training counter.
        """
        self.train_step += 1

    @staticmethod
    def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        """
        Soft update target network:
            target = tau * source + (1 - tau) * target
        """
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * source_param.data)

    @staticmethod
    def hard_update(source: torch.nn.Module, target: torch.nn.Module) -> None:
        """
        Directly copy source network weights into target network.
        """
        target.load_state_dict(source.state_dict())

    def _set_seed(self, seed: Optional[int]) -> None:
        """
        Set random seeds for reproducibility.
        """
        if seed is None:
            return

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def config_dict(self) -> Dict[str, Any]:
        """
        Return config as dictionary.
        """
        return self.config.__dict__.copy()
