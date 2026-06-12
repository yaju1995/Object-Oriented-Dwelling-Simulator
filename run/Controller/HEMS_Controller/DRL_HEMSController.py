import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Type, Optional

from run.Controller.HEMS_Controller.controller_HEMS import HEMSController
from run.Controller.deep_rl.drl_agent_handler import DRLAgent, DRLAgentConfig


@dataclass
class DRLControllerConfig:
    agent_class: Type[DRLAgent]
    agent_config: DRLAgentConfig


class DRLController(HEMSController):
    def __init__(
            self, name, resolution, tariff_info, train, update_period, controller_config: DRLControllerConfig, ):
        super().__init__(name, resolution, tariff_info, train, update_period)

        self.controller_config = controller_config

        self.agent: DRLAgent = controller_config.agent_class(
            controller_config.agent_config
        )

        self.state = None
        self.action = None
        self.next_state = None

        # Evalution parameter
        self.avg_reward = 0
        self.cumulative_reward = 0
        self.eps_count = 0

    def control_logic(self, done: bool, *args, **kwargs):
        if self.state is not None and self.action is not None:
            reward = self.get_reward()
            self.eps_count += 1
            self.cumulative_reward += reward

            self.next_state = self.get_state()

            print(self.state, self.action, reward, self.next_state, done)
            self.agent.store_transition(self.state, self.action, reward, self.next_state, done)

            if self.train:
                self.agent.train()

        if done:
            self.state = None
            self.action = None
            self.next_state = None
            self.action = 0
            return

        if self.next_state is not None:
            self.state = self.next_state
        else:
            self.state = self.get_state()

        raw_action = self.agent.get_action(self.state)
        self.action = raw_action

        return

    @abstractmethod
    def get_observation(self) -> np.ndarray | list | dict:
        pass

    @abstractmethod
    def get_reward(self) -> int:
        return 0

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    def save_models(self, path: str = None):
        self.agent.save(path)

    def load_models(self, path: str = None):
        self.agent.load(path)
