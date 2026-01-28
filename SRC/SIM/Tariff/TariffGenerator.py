import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np


class BaseTariffGenerator(ABC):
    """
    Abstract parent class for tariff generators.
    Subclasses must implement `generate_tariff()`.
    """

    def __init__(self, resolution: timedelta = timedelta(minutes=1)):
        """
        Parameters
        ----------
        resolution : timedelta
            Time step between tariff entries.
        """
        self.resolution = resolution

    @property
    def time_index(self):
        """Return a 24h index as datetime.time objects using timedelta steps."""
        times = []
        current = timedelta(0)
        end = timedelta(days=1)

        while current < end:
            # Convert timedelta to a time object
            t = (datetime.min + current).time()
            times.append(t)
            current += self.resolution

        return times

    @abstractmethod
    def generate_tariff(self) -> pd.DataFrame:
        """
        Must return a DataFrame with:
            index = datetime.time
            column = 'value'
        """
        pass


class UniformTariffGenerator(BaseTariffGenerator):
    def __init__(self, value: float, resolution: timedelta = timedelta(minutes=1)):
        super().__init__(resolution)
        self.value = value

    def generate_tariff(self) -> pd.DataFrame:
        idx = self.time_index
        return pd.DataFrame({"value": self.value}, index=idx)


class RandomTariffGenerator(BaseTariffGenerator):
    def __init__(self, low: float, high: float,
                 resolution: timedelta = timedelta(minutes=1)):
        super().__init__(resolution)
        self.low = low
        self.high = high

    def generate_tariff(self) -> pd.DataFrame:
        idx = self.time_index
        values = np.round(np.random.uniform(self.low, self.high, size=len(idx)),3)
        return pd.DataFrame({"value": values}, index=idx)


if __name__ == '__main__':
    # gen = UniformTariffGenerator(value=0.12, resolution=timedelta(minutes=60))
    gen = RandomTariffGenerator(low=0.1, high=0.5, resolution=timedelta(minutes=60))
    tariff = gen.generate_tariff()
    print(tariff)
