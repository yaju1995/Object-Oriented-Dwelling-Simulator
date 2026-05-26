from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class NumpyDataStore:
    """
    Fast replacement for a pandas-row-append DataStore.

    Runtime storage:
        NumPy arrays

    Compatibility output:
        pandas DataFrame / Series

    Same public functions:
        append(...)
        get_past_period_df(...)
        get_resampled(...)
        past_period_resampled(...)
        get_instant_state(...)
        to_dataframe(...)
    """

    resolution: timedelta
    columns: list[str] | None = None
    maxlen: int | None = None
    dtype: Any = np.float64
    lock: RLock = field(default_factory=RLock)

    _times: np.ndarray | None = field(default=None, init=False, repr=False)
    _data: np.ndarray | None = field(default=None, init=False, repr=False)
    _columns: list[str] = field(default_factory=list, init=False)
    _col_idx: dict[str, int] = field(default_factory=dict, init=False)

    _start: int = field(default=0, init=False)
    _size: int = field(default=0, init=False)
    _capacity: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.columns is not None:
            self._columns = list(self.columns)
            self._col_idx = {c: i for i, c in enumerate(self._columns)}

        if self.maxlen is not None and self.maxlen <= 0:
            raise ValueError("maxlen must be positive or None")

        initial_capacity = self.maxlen if self.maxlen is not None else 1024
        self._allocate(initial_capacity)

    @property
    def df(self) -> pd.DataFrame:
        """
        Compatibility with old code that uses database.df.

        Warning:
            This builds a DataFrame on demand.
            Use it for export/logging, not inside every step.
        """
        return self.to_dataframe()

    def _allocate(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._times = np.empty(self._capacity, dtype="datetime64[ns]")
        self._times[:] = np.datetime64("NaT")

        n_cols = len(self._columns)
        self._data = np.full((self._capacity, n_cols), np.nan, dtype=self.dtype)

    def _ensure_columns(self, row: dict[str, Any]) -> None:
        new_cols = [c for c in row.keys() if c not in self._col_idx]
        if not new_cols:
            return

        self._columns.extend(new_cols)
        self._col_idx = {c: i for i, c in enumerate(self._columns)}

        extra = np.full((self._capacity, len(new_cols)), np.nan, dtype=self.dtype)
        self._data = np.hstack([self._data, extra])

    def _ordered_indices(self) -> np.ndarray:
        if self._size == 0:
            return np.array([], dtype=np.int64)

        if self.maxlen is None or self._start == 0:
            return np.arange(self._size)

        return (self._start + np.arange(self._size)) % self._capacity

    def _ordered_times(self) -> np.ndarray:
        idx = self._ordered_indices()
        return self._times[idx].copy()

    def _ordered_data(self) -> np.ndarray:
        idx = self._ordered_indices()
        return self._data[idx, :].copy()

    def _grow_if_needed(self) -> None:
        if self.maxlen is not None:
            return

        if self._size < self._capacity:
            return

        old_times = self._ordered_times()
        old_data = self._ordered_data()

        old_size = self._size
        new_capacity = max(2 * self._capacity, 1024)
        self._allocate(new_capacity)

        self._times[:old_size] = old_times
        self._data[:old_size, :] = old_data

        self._start = 0
        self._size = old_size

    def _write_position(self) -> int:
        if self.maxlen is None:
            self._grow_if_needed()
            pos = self._size
            self._size += 1
            return pos

        if self._size < self._capacity:
            pos = (self._start + self._size) % self._capacity
            self._size += 1
            return pos

        pos = self._start
        self._start = (self._start + 1) % self._capacity
        return pos

    def append(self, t: datetime | pd.Timestamp, row: dict[str, Any]) -> None:
        """
        Append one row.

        This assumes timestamps are appended in increasing simulation order.
        It does not sort every step, which is the main speed improvement.
        """
        t64 = np.datetime64(pd.Timestamp(t).to_datetime64())

        with self.lock:
            self._ensure_columns(row)
            pos = self._write_position()

            self._times[pos] = t64
            self._data[pos, :] = np.nan

            for key, value in row.items():
                col = self._col_idx[key]
                try:
                    self._data[pos, col] = value
                except (TypeError, ValueError):
                    self._data[pos, col] = np.nan

    def get_past_period_df(
        self,
        now_time: datetime | pd.Timestamp,
        past_period: int = 24,
    ) -> pd.DataFrame | None:
        """
        Return exactly N samples ending at the latest timestamp <= now_time.

        N = past_period(hours) / self.resolution
        """
        now64 = np.datetime64(pd.Timestamp(now_time).to_datetime64())
        res_pd = pd.Timedelta(self.resolution)
        period_pd = pd.Timedelta(hours=past_period)

        expected_len = int(period_pd.total_seconds() / res_pd.total_seconds())
        if expected_len <= 0:
            raise ValueError("past_period too small for the configured resolution")

        with self.lock:
            if self._size == 0:
                return None

            times = self._ordered_times()
            data = self._ordered_data()

        pos = np.searchsorted(times, now64, side="right") - 1
        if pos < 0:
            return None

        start_pos = pos - expected_len + 1
        if start_pos < 0:
            return None

        window_times = times[start_pos:pos + 1]
        window_data = data[start_pos:pos + 1, :]

        if len(window_times) != expected_len:
            return None

        end_time = pd.Timestamp(window_times[-1])
        expected_index = pd.date_range(
            end=end_time,
            periods=expected_len,
            freq=res_pd,
            name="time",
        )
        actual_index = pd.DatetimeIndex(window_times, name="time")

        if not actual_index.equals(expected_index):
            return None

        return pd.DataFrame(
            window_data.copy(),
            index=actual_index,
            columns=self._columns,
        )

    def get_resampled(
        self,
        df_sample: pd.DataFrame,
        resolution: timedelta,
        headers: list[str],
        agg: str = "mean",
    ) -> pd.DataFrame | None:
        """
        Resample a small returned history window.
        """
        if df_sample is None or df_sample.empty:
            return None

        if not isinstance(df_sample.index, pd.DatetimeIndex):
            raise ValueError("df_sample must have a DatetimeIndex")

        missing = set(headers) - set(df_sample.columns)
        if missing:
            raise KeyError(f"Missing columns in df_sample: {missing}")

        rule = pd.Timedelta(resolution)
        r = df_sample[headers].resample(rule)

        if agg == "mean":
            return r.mean()
        if agg == "sum":
            return r.sum()
        if agg == "max":
            return r.max()
        if agg == "min":
            return r.min()

        raise ValueError(f"Unsupported aggregation: {agg}")

    def past_period_resampled(
        self,
        now_time: datetime | pd.Timestamp,
        past_period: int,
        out_resolution: timedelta,
        headers: list[str],
        agg: str = "mean",
    ) -> pd.DataFrame | None:
        w = self.get_past_period_df(now_time, past_period)
        if w is None:
            return None
        return self.get_resampled(w, out_resolution, headers, agg)

    def get_instant_state(
        self,
        keys: list[str],
        *,
        now_time: datetime | pd.Timestamp | None = None,
        strict: bool = True,
    ) -> pd.Series | None:
        """
        Return the latest values for selected keys.

        Output is a pandas Series for compatibility.
        """
        with self.lock:
            if self._size == 0:
                return None

            missing = set(keys) - set(self._columns)
            if missing and strict:
                raise KeyError(f"Missing instant keys: {missing}")

            times = self._ordered_times()
            data = self._ordered_data()

        if now_time is None:
            pos = len(times) - 1
        else:
            now64 = np.datetime64(pd.Timestamp(now_time).to_datetime64())
            pos = np.searchsorted(times, now64, side="right") - 1
            if pos < 0:
                return None

        values = {}
        for key in keys:
            if key in self._col_idx:
                values[key] = data[pos, self._col_idx[key]]
            else:
                values[key] = np.nan

        return pd.Series(values, name=pd.Timestamp(times[pos]))

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export stored data to pandas.
        """
        with self.lock:
            if self._size == 0:
                return pd.DataFrame(columns=self._columns)

            times = self._ordered_times()
            data = self._ordered_data()

        return pd.DataFrame(
            data,
            index=pd.DatetimeIndex(times, name="time"),
            columns=self._columns,
        )

    def clear(self) -> None:
        with self.lock:
            self._start = 0
            self._size = 0
            self._times[:] = np.datetime64("NaT")
            self._data[:, :] = np.nan

    def __len__(self) -> int:
        return self._size
