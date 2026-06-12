from abc import ABC, abstractmethod
from datetime import timedelta


class BaseController(ABC):
    def __init__(
        self,
        name: str,
    ):
        self.name = name
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def control_logic(self, *args, **kwargs):
        pass

    # @abstractmethod
    def save_models(self, *args, **kwargs):
        pass

    # @abstractmethod
    def load_models(self,  *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass