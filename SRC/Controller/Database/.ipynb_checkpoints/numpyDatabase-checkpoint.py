from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

import numpy as np


@dataclass
class DataStore:
    """
    Lightweight NumPy-based time-series database.

    Designed for fast append during simulation and instant-state lookup.

    Assumptions:
    - Data is appended mostly in chronological order.
    - No resampling is required.
    - Each row is a dict of scalar values.
    """

    resolution: timedelta
    initial_capacity: int = 10_000

    lock: RLock = field(default_factory=RLock, init=False)

    # Internal storage
    times: np.ndarray = field(init=False)
    data: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    size: int = field(default=0, init=False)
    capacity: int = field(init=False)

    def __post_init__(self):
        self.capacity = self.initial_capacity

        self.times = np.empty(self.capacity, dtype="datetime64[us]")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _to_np_time(self, t: datetime | np.datetime64) -> np.datetime64:
        return np.datetime64(t, "us")

    def _ensure_capacity(self, required_size: int) -> None:
        if required_size <= self.capacity:
            return

        new_capacity = max(required_size, self.capacity * 2)

        new_times = np.empty(new_capacity, dtype="datetime64[us]")
        new_times[: self.size] = self.times[: self.size]
        self.times = new_times

        for key, arr in self.data.items():
            new_arr = np.full(new_capacity, np.nan, dtype=arr.dtype)
            new_arr[: self.size] = arr[: self.size]
            self.data[key] = new_arr

        self.capacity = new_capacity

    def _ensure_column(self, key: str, value: Any) -> None:
        if key in self.data:
            return

        dtype = self._infer_dtype(value)

        if np.issubdtype(dtype, np.number):
            arr = np.full(self.capacity, np.nan, dtype=dtype)
        else:
            arr = np.empty(self.capacity, dtype=object)
            arr[:] = None

        self.data[key] = arr

    def _infer_dtype(self, value: Any):
        if isinstance(value, bool):
            return np.float32

        if isinstance(value, int):
            return np.float32

        if isinstance(value, float):
            return np.float32

        if isinstance(value, np.number):
            return np.float32

        return object

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------
    def append(self, t: datetime | np.datetime64, row: dict[str, Any]) -> None:
        """
        Append one simulation row.

        Example:
            db.append(now_time, {
                "Battery SOC (-)": 0.52,
                "Total Electric Power (kW)": 1.4,
            })
        """

        t_np = self._to_np_time(t)

        with self.lock:
            self._ensure_capacity(self.size + 1)

            idx = self.size
            self.times[idx] = t_np

            # Create missing columns
            for key, value in row.items():
                self._ensure_column(key, value)

            # Fill all existing columns with default missing value
            for key, arr in self.data.items():
                if arr.dtype == object:
                    arr[idx] = None
                else:
                    arr[idx] = np.nan

            # Write provided values
            for key, value in row.items():
                self.data[key][idx] = value

            self.size += 1

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    def get_instant_state(
        self,
        keys: list[str],
        *,
        now_time: datetime | np.datetime64 | None = None,
        strict: bool = True,
    ) -> dict[str, Any] | None:
        """
        Return latest values for selected keys.

        If now_time is provided:
            returns the latest row where time <= now_time.

        If now_time is None:
            returns the latest appended row.

        Returns:
            dict[str, Any] or None
        """

        with self.lock:
            if self.size == 0:
                return None

            # Validate keys
            missing = set(keys) - set(self.data.keys())

            if missing and strict:
                raise KeyError(f"Missing instant keys: {missing}")

            if now_time is None:
                idx = self.size - 1
            else:
                t_np = self._to_np_time(now_time)

                valid_times = self.times[: self.size]

                # equivalent to pandas get_indexer(method="pad")
                idx = np.searchsorted(valid_times, t_np, side="right") - 1

                if idx < 0:
                    return None

            result = {}

            for key in keys:
                if key in self.data:
                    result[key] = self.data[key][idx]
                else:
                    result[key] = np.nan

            return result

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------
    def columns(self) -> list[str]:
        return list(self.data.keys())

    def latest_time(self) -> np.datetime64 | None:
        if self.size == 0:
            return None
        return self.times[self.size - 1]

    def clear(self) -> None:
        with self.lock:
            self.size = 0

    def to_dict(self) -> dict[str, np.ndarray]:
        """
        Return trimmed NumPy arrays.
        """
        with self.lock:
            out = {
                "time": self.times[: self.size].copy()
            }

            for key, arr in self.data.items():
                out[key] = arr[: self.size].copy()

            return out