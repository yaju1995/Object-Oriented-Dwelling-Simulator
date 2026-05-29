import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime, date
from typing import Optional, Union

from SRC.support.lib_config import CustomLogger
from SRC.SIM.Tariff.TariffGenerator import BaseTariffGenerator

logger = CustomLogger(command=False)


# =============================================================================
# Fast time helpers
# =============================================================================

def time_to_seconds(t) -> int:
    """
    Accepts datetime.time, datetime.datetime, date-like pandas Timestamp,
    and returns seconds since midnight.
    """
    if isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    if isinstance(t, datetime):
        return t.hour * 3600 + t.minute * 60 + t.second
    if isinstance(t, dtime):
        return t.hour * 3600 + t.minute * 60 + t.second
    raise TypeError(f"Unsupported type for time_to_seconds: {type(t)}")


def _seconds_from_time_index(index) -> np.ndarray:
    """
    Convert a pandas index of datetime.time values into sorted seconds-of-day.
    """
    return np.asarray([time_to_seconds(t) for t in index], dtype=np.int32)


def get_active_period_value(df: pd.DataFrame, now_time) -> float:
    """
    Backward-compatible standalone function.

    df index: datetime.time values.
    df must contain a column named 'value'.

    This keeps the old public behaviour but uses NumPy searchsorted internally.
    """
    if df is None or df.empty:
        raise ValueError("Tariff DataFrame is None or empty.")
    if "value" not in df.columns:
        raise ValueError("Tariff DataFrame must contain a 'value' column.")

    idx_seconds = _seconds_from_time_index(df.index)
    values = df["value"].to_numpy(dtype=float, copy=False)
    now_seconds = time_to_seconds(now_time)

    pos = np.searchsorted(idx_seconds, now_seconds, side="right") - 1

    if pos < 0:
        # Before first period: previous day's last period.
        pos = len(values) - 1

    return float(values[pos])


# =============================================================================
# Tariff handler
# =============================================================================

class tariffHandler:
    """
    Tariff handler with pandas-compatible public API and NumPy-based runtime lookup.

    Main idea:
      - Keep existing functions and DataFrame attributes:
            self.tariff
            self.feed_tariff
            self.next_24hr_tariff
            self.next_24hr_feed_tariff
      - Add internal NumPy caches:
            self._tariff_seconds
            self._tariff_values
            self._feed_seconds
            self._feed_values
            self._next_tariff_seconds
            self._next_tariff_values
            self._next_feed_seconds
            self._next_feed_values

    Use these fast methods in your controller/simulator loop:
      - get_tariff(...)
      - get_tariff_range_array(...)

    Use this only for debug/export:
      - get_tariff_range_df(...)
    """

    def __init__(
        self,
        tariff_model: Optional[BaseTariffGenerator] = None,
        feed_tariff_model: Optional[BaseTariffGenerator] = None,
        type: int = 2,
        tariff_resolution: timedelta = timedelta(minutes=60),
    ):
        """
        type:
          1 -> same tariff for import/export
          2 -> separate import tariff + feed-in tariff
        """
        self.tariff_model = tariff_model
        self.feed_tariff_model = feed_tariff_model
        self.tariff_resolution = tariff_resolution
        self.type = type

        # Active current-day tariff profiles, kept for compatibility.
        self.tariff: Optional[pd.DataFrame] = None
        self.feed_tariff: Optional[pd.DataFrame] = None

        # Staged next-day tariff profiles, kept for compatibility.
        self.next_24hr_tariff: Optional[pd.DataFrame] = None
        self.next_24hr_feed_tariff: Optional[pd.DataFrame] = None

        # Historic stores: {date: DataFrame(index=time, columns=["value"])}
        self.historic_tariff_by_day: dict[date, pd.DataFrame] = {}
        self.historic_feed_tariff_by_day: dict[date, pd.DataFrame] = {}
        self.historic_tariff_resolution: Optional[timedelta] = None
        self.historic_feed_tariff_resolution: Optional[timedelta] = None

        # Fast NumPy caches for current and next-day profiles.
        self._tariff_seconds: Optional[np.ndarray] = None
        self._tariff_values: Optional[np.ndarray] = None
        self._feed_seconds: Optional[np.ndarray] = None
        self._feed_values: Optional[np.ndarray] = None

        self._next_tariff_seconds: Optional[np.ndarray] = None
        self._next_tariff_values: Optional[np.ndarray] = None
        self._next_feed_seconds: Optional[np.ndarray] = None
        self._next_feed_values: Optional[np.ndarray] = None

        # Combined 48h cache for fast range queries.
        self._cache_48_valid = False
        self._tariff_48_seconds: Optional[np.ndarray] = None
        self._tariff_48_values: Optional[np.ndarray] = None
        self._feed_48_seconds: Optional[np.ndarray] = None
        self._feed_48_values: Optional[np.ndarray] = None

        # Stats.
        self.max_tariff: Optional[float] = None
        self.min_tariff: Optional[float] = None
        self.max_feed_tariff: Optional[float] = None
        self.min_feed_tariff: Optional[float] = None
        self.avg_feed_tariff: Optional[float] = None
        self.avg_tariff: Optional[float] = None

    # -------------------------------------------------------------------------
    # Internal cache helpers
    # -------------------------------------------------------------------------
    def _invalidate_48h_cache(self) -> None:
        self._cache_48_valid = False
        self._tariff_48_seconds = None
        self._tariff_48_values = None
        self._feed_48_seconds = None
        self._feed_48_values = None

    def _df_to_arrays(self, df: Optional[pd.DataFrame]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert a time-indexed DataFrame with a 'value' column into:
            seconds_of_day, values

        The result is sorted by seconds-of-day and duplicate times keep the last value.
        """
        if df is None or df.empty:
            return None, None

        if "value" not in df.columns:
            raise ValueError("Tariff DataFrame must contain a 'value' column.")

        # Ensure stable sorted unique time index.
        work = df[["value"]].copy()
        work = work[~work.index.duplicated(keep="last")]
        work = work.sort_index()

        seconds = _seconds_from_time_index(work.index)
        values = work["value"].to_numpy(dtype=np.float64, copy=True)

        order = np.argsort(seconds)
        seconds = seconds[order]
        values = values[order]

        return seconds.astype(np.int32, copy=False), values.astype(np.float64, copy=False)

    def _sync_current_arrays(self) -> None:
        self._tariff_seconds, self._tariff_values = self._df_to_arrays(self.tariff)

        if self.type == 1:
            self.feed_tariff = self.tariff.copy() if self.tariff is not None else None
            self._feed_seconds = self._tariff_seconds
            self._feed_values = self._tariff_values
        else:
            self._feed_seconds, self._feed_values = self._df_to_arrays(self.feed_tariff)

        self._invalidate_48h_cache()

    def _sync_next_arrays(self) -> None:
        self._next_tariff_seconds, self._next_tariff_values = self._df_to_arrays(self.next_24hr_tariff)

        if self.type == 1:
            self.next_24hr_feed_tariff = (
                self.next_24hr_tariff.copy()
                if self.next_24hr_tariff is not None
                else None
            )
            self._next_feed_seconds = self._next_tariff_seconds
            self._next_feed_values = self._next_tariff_values
        else:
            self._next_feed_seconds, self._next_feed_values = self._df_to_arrays(
                self.next_24hr_feed_tariff
            )

        self._invalidate_48h_cache()

    def _sync_all_arrays(self) -> None:
        self._sync_current_arrays()
        self._sync_next_arrays()

    def _lookup_from_arrays(
        self,
        seconds: Optional[np.ndarray],
        values: Optional[np.ndarray],
        now_time,
        default: float = 0.0,
        *,
        wrap_before_first: bool = True,
    ) -> float:
        """
        Fast active-period lookup.

        If wrap_before_first=True:
            before first timestamp -> previous day's last tariff value.
        """
        if seconds is None or values is None or len(values) == 0:
            return float(default)

        now_seconds = time_to_seconds(now_time)
        pos = np.searchsorted(seconds, now_seconds, side="right") - 1

        if pos < 0:
            if wrap_before_first:
                pos = len(values) - 1
            else:
                pos = 0

        return float(values[pos])

    def _build_48h_arrays(self) -> None:
        """
        Build cached 48h arrays:
          - current day: offsets 0..86399
          - next day:    offsets 86400..172799

        If next-day data is not staged, current-day data is repeated.
        """
        if self._cache_48_valid:
            return

        # Import tariff.
        cur_s = self._tariff_seconds
        cur_v = self._tariff_values

        next_s = self._next_tariff_seconds if self._next_tariff_seconds is not None else cur_s
        next_v = self._next_tariff_values if self._next_tariff_values is not None else cur_v

        if cur_s is None or cur_v is None:
            self._tariff_48_seconds = np.empty(0, dtype=np.int32)
            self._tariff_48_values = np.empty(0, dtype=np.float64)
        else:
            if next_s is None or next_v is None:
                next_s, next_v = cur_s, cur_v

            self._tariff_48_seconds = np.concatenate(
                (cur_s, next_s + 86400)
            ).astype(np.int32, copy=False)
            self._tariff_48_values = np.concatenate(
                (cur_v, next_v)
            ).astype(np.float64, copy=False)

        # Feed tariff.
        feed_s = self._feed_seconds
        feed_v = self._feed_values

        next_feed_s = self._next_feed_seconds if self._next_feed_seconds is not None else feed_s
        next_feed_v = self._next_feed_values if self._next_feed_values is not None else feed_v

        if self.type == 1:
            self._feed_48_seconds = self._tariff_48_seconds
            self._feed_48_values = self._tariff_48_values
        elif feed_s is None or feed_v is None:
            self._feed_48_seconds = np.empty(0, dtype=np.int32)
            self._feed_48_values = np.empty(0, dtype=np.float64)
        else:
            if next_feed_s is None or next_feed_v is None:
                next_feed_s, next_feed_v = feed_s, feed_v

            self._feed_48_seconds = np.concatenate(
                (feed_s, next_feed_s + 86400)
            ).astype(np.int32, copy=False)
            self._feed_48_values = np.concatenate(
                (feed_v, next_feed_v)
            ).astype(np.float64, copy=False)

        self._cache_48_valid = True

    def _range_from_48h_arrays(
        self,
        seconds_48: Optional[np.ndarray],
        values_48: Optional[np.ndarray],
        start_second: int,
        period: int,
        step_seconds: int,
        default: float = 0.0,
    ) -> np.ndarray:
        """
        Vectorized forward-fill lookup over a 48h tariff profile.
        """
        if seconds_48 is None or values_48 is None or len(values_48) == 0:
            return np.full(period, default, dtype=np.float64)

        query_seconds = start_second + np.arange(period, dtype=np.int32) * step_seconds

        # Clamp to 48h window if a very long lookahead is requested.
        # For periods beyond 48h, we repeat the last known value.
        query_seconds = np.minimum(query_seconds, 172799)

        pos = np.searchsorted(seconds_48, query_seconds, side="right") - 1

        # This mimics old get_tariff_range_df behaviour: reindex(ffill), then bfill.
        # So if the query is before the first tariff period, use the first known value.
        pos[pos < 0] = 0

        return values_48[pos].astype(np.float64, copy=False)

    # -------------------------------------------------------------------------
    # Generator-based tariff handling
    # -------------------------------------------------------------------------
    def generate_tariff(self) -> None:
        """
        Generates next-day/next-24h tariffs into the staging buffers.
        """
        if self.tariff_model is not None:
            self.next_24hr_tariff = self.tariff_model.generate_tariff()

        if self.feed_tariff_model is not None:
            self.next_24hr_feed_tariff = self.feed_tariff_model.generate_tariff()

        if self.type == 1 and self.next_24hr_tariff is not None:
            self.next_24hr_feed_tariff = self.next_24hr_tariff.copy()

        self._sync_next_arrays()

    # -------------------------------------------------------------------------
    # Standard single-day tariff CSV upload
    # Expected format:
    #   index = HH:MM:SS
    #   columns include 'value'
    # -------------------------------------------------------------------------
    def _load_tariff_csv(self, file: str) -> pd.DataFrame:
        df = pd.read_csv(file, index_col=0)

        if "value" not in df.columns:
            raise ValueError(
                f"Tariff file '{file}' must contain a 'value' column. "
                f"Found: {list(df.columns)}"
            )

        idx = pd.to_datetime(df.index, format="%H:%M:%S", errors="raise").time
        df.index = idx
        df = df.sort_index()
        return df

    def upload_tariff(self, file: str) -> None:
        self.tariff = self._load_tariff_csv(file)
        if self.type == 1:
            self.feed_tariff = self.tariff.copy()

        self._sync_current_arrays()
        self.update_min_max()

    def upload_feed_tariff(self, file: str) -> None:
        self.feed_tariff = self._load_tariff_csv(file)

        if self.type == 1:
            self.feed_tariff = self.tariff.copy() if self.tariff is not None else None

        self._sync_current_arrays()
        self.update_min_max()

    # -------------------------------------------------------------------------
    # Historic tariff CSV upload
    # Expected format like Irish_2026_Wholesale_price.csv:
    #   Timestamp, Euro/kWh
    # -------------------------------------------------------------------------
    def _load_historic_tariff_csv(
        self,
        file: str,
        timestamp_col: str = "Timestamp",
        value_col: str = "Euro/kWh",
        dayfirst: bool = True,
    ) -> tuple[dict[date, pd.DataFrame], Optional[timedelta]]:
        """
        Loads a timestamped historic tariff CSV and arranges it by date:
            {
                date(2026, 1, 1): DataFrame(index=time, columns=["value"]),
                date(2026, 1, 2): DataFrame(index=time, columns=["value"]),
                ...
            }
        """
        df = pd.read_csv(file)

        if timestamp_col not in df.columns:
            raise ValueError(
                f"Missing timestamp column '{timestamp_col}'. Found: {list(df.columns)}"
            )
        if value_col not in df.columns:
            raise ValueError(
                f"Missing value column '{value_col}'. Found: {list(df.columns)}"
            )

        df = df[[timestamp_col, value_col]].copy()
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col],
            errors="raise",
            dayfirst=dayfirst,
        )
        df = df.rename(columns={value_col: "value"})
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        diffs = df[timestamp_col].diff().dropna()
        resolution = None if diffs.empty else diffs.median().to_pytimedelta()

        df["date"] = df[timestamp_col].dt.date
        df["time"] = df[timestamp_col].dt.time

        by_day: dict[date, pd.DataFrame] = {}
        for day_key, g in df.groupby("date", sort=True):
            day_df = g[["time", "value"]].copy()
            day_df = day_df.drop_duplicates(subset="time", keep="last")
            day_df = day_df.set_index("time").sort_index()
            by_day[day_key] = day_df

        return by_day, resolution

    def upload_historic_tariff(
        self,
        file: str,
        timestamp_col: str = "Timestamp",
        value_col: str = "Euro/kWh",
        dayfirst: bool = True,
    ) -> None:
        """
        Upload historic import tariff data and arrange it by day.
        """
        self.historic_tariff_by_day, self.historic_tariff_resolution = self._load_historic_tariff_csv(
            file=file,
            timestamp_col=timestamp_col,
            value_col=value_col,
            dayfirst=dayfirst,
        )

    def upload_historic_feed_tariff(
        self,
        file: str,
        timestamp_col: str = "Timestamp",
        value_col: str = "Euro/kWh",
        dayfirst: bool = True,
    ) -> None:
        """
        Upload historic feed tariff data and arrange it by day.
        """
        self.historic_feed_tariff_by_day, self.historic_feed_tariff_resolution = self._load_historic_tariff_csv(
            file=file,
            timestamp_col=timestamp_col,
            value_col=value_col,
            dayfirst=dayfirst,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _as_time(self, dt_like) -> dtime:
        """
        Return time-of-day from datetime/pd.Timestamp/time.
        """
        if isinstance(dt_like, pd.Timestamp):
            dt_like = dt_like.to_pydatetime()
        if isinstance(dt_like, datetime):
            return dt_like.time()
        if isinstance(dt_like, dtime):
            return dt_like
        raise TypeError(f"Unsupported time type: {type(dt_like)}")

    def _as_date(self, dt_like) -> date:
        """
        Return date from datetime/pd.Timestamp/date.
        """
        if isinstance(dt_like, pd.Timestamp):
            return dt_like.date()
        if isinstance(dt_like, datetime):
            return dt_like.date()
        if isinstance(dt_like, date):
            return dt_like
        raise TypeError(f"Unsupported date type: {type(dt_like)}")

    def _stamp_day_tariff(self, day_date, day_tariff_df: pd.DataFrame) -> pd.Series:
        """
        Convert a time-index tariff DF into a datetime-index series for a given date.
        Kept for backward compatibility/debug use.
        """
        if day_tariff_df is None or day_tariff_df.empty:
            return pd.Series(dtype=float)

        if "value" not in day_tariff_df.columns:
            raise ValueError("Tariff DataFrame must have a 'value' column")

        dt_index = pd.to_datetime([datetime.combine(day_date, t) for t in day_tariff_df.index])
        s = pd.Series(day_tariff_df["value"].astype(float).values, index=dt_index)
        s = s.sort_index()
        return s

    def _build_48h_tariff_series(self, now_time: datetime, which: str) -> pd.Series:
        """
        Backward-compatible pandas 48h tariff series builder.
        Runtime code should prefer get_tariff_range_array().
        """
        base_date = pd.Timestamp(now_time).date()
        tomorrow = (pd.Timestamp(now_time) + pd.Timedelta(days=1)).date()

        if which == "tariff":
            today_df = self.tariff
            tomorrow_df = self.next_24hr_tariff if self.next_24hr_tariff is not None else self.tariff
        elif which == "feed_tariff":
            today_df = self.feed_tariff
            tomorrow_df = (
                self.next_24hr_feed_tariff
                if self.next_24hr_feed_tariff is not None
                else self.feed_tariff
            )
        else:
            raise ValueError("which must be 'tariff' or 'feed_tariff'")

        s_today = self._stamp_day_tariff(base_date, today_df)
        s_tom = self._stamp_day_tariff(tomorrow, tomorrow_df)

        s_48 = pd.concat([s_today, s_tom]).sort_index()
        return s_48

    def validate_historic_day(
        self,
        target_date,
        which: str = "tariff",
        expected_steps: Optional[int] = None,
    ) -> bool:
        """
        Check if a historic day exists and optionally whether it has the expected
        number of timesteps.
        """
        target_date = self._as_date(target_date)

        if which == "tariff":
            store = self.historic_tariff_by_day
            resolution = self.historic_tariff_resolution
        elif which == "feed_tariff":
            store = self.historic_feed_tariff_by_day
            resolution = self.historic_feed_tariff_resolution
        else:
            raise ValueError("which must be 'tariff' or 'feed_tariff'")

        if target_date not in store:
            return False

        day_df = store[target_date]

        if expected_steps is None and resolution is not None:
            expected_steps = int(timedelta(days=1) / resolution)

        if expected_steps is None:
            return True

        return len(day_df) == expected_steps

    # -------------------------------------------------------------------------
    # Historic-data driven current/next day update
    # -------------------------------------------------------------------------
    def set_current_tariff_from_historic(self, target_date, update_feed: bool = True) -> None:
        """
        Set self.tariff and optionally self.feed_tariff from historic data.
        """
        target_date = self._as_date(target_date)

        if not self.historic_tariff_by_day:
            raise ValueError("No historic import tariff data uploaded.")
        if target_date not in self.historic_tariff_by_day:
            raise KeyError(f"No historic import tariff found for {target_date}")

        self.tariff = self.historic_tariff_by_day[target_date].copy()

        if update_feed:
            if self.type == 1:
                self.feed_tariff = self.tariff.copy()
            elif self.historic_feed_tariff_by_day:
                if target_date not in self.historic_feed_tariff_by_day:
                    raise KeyError(f"No historic feed tariff found for {target_date}")
                self.feed_tariff = self.historic_feed_tariff_by_day[target_date].copy()

        self._sync_current_arrays()
        self.update_min_max()

    def update_next_24hr_from_historic(self, target_date, update_feed: bool = True) -> None:
        """
        Load a specific day's tariff from historic data into the next-day staging buffers.
        """
        target_date = self._as_date(target_date)

        if not self.historic_tariff_by_day:
            raise ValueError("No historic import tariff data uploaded.")
        if target_date not in self.historic_tariff_by_day:
            raise KeyError(f"No historic import tariff found for {target_date}")

        self.next_24hr_tariff = self.historic_tariff_by_day[target_date].copy()

        if update_feed:
            if self.type == 1:
                self.next_24hr_feed_tariff = self.next_24hr_tariff.copy()
            elif self.historic_feed_tariff_by_day:
                if target_date not in self.historic_feed_tariff_by_day:
                    raise KeyError(f"No historic feed tariff found for {target_date}")
                self.next_24hr_feed_tariff = self.historic_feed_tariff_by_day[target_date].copy()

        self._sync_next_arrays()

    def prepare_day_ahead_tariffs(self, now_time, update_feed: bool = True) -> None:
        """
        For a given simulator timestamp:
          - self.tariff = tariff for current date
          - self.next_24hr_tariff = tariff for next date
        """
        now_ts = pd.Timestamp(now_time)
        today = now_ts.date()
        tomorrow = (now_ts + pd.Timedelta(days=1)).date()

        self.set_current_tariff_from_historic(today, update_feed=update_feed)
        self.update_next_24hr_from_historic(tomorrow, update_feed=update_feed)

    # -------------------------------------------------------------------------
    # Tariff access
    # -------------------------------------------------------------------------
    def get_tariff(self, now_time) -> tuple[float, float]:
        """
        Fast NumPy lookup.

        Returns:
            (import_tariff, feed_in_tariff)
        """
        tariff = self._lookup_from_arrays(
            self._tariff_seconds,
            self._tariff_values,
            now_time,
            default=0.0,
            wrap_before_first=True,
        )

        if self.type == 1:
            return float(tariff), float(tariff)

        feed_tariff = self._lookup_from_arrays(
            self._feed_seconds,
            self._feed_values,
            now_time,
            default=0.0,
            wrap_before_first=True,
        )

        return float(tariff), float(feed_tariff)

    def get_tariff_range_array(
        self,
        now_time: datetime,
        period: int = 96,
        resolution: timedelta = timedelta(minutes=15),
        copy: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast NumPy version of get_tariff_range_df().

        Returns:
            tariff_values, feed_tariff_values

        This is the method to use inside get_state()/RL loops.
        """
        self._build_48h_arrays()

        start_second = time_to_seconds(now_time)
        step_seconds = int(resolution.total_seconds())

        if step_seconds <= 0:
            raise ValueError("resolution must be positive.")
        if period <= 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        tariff_vals = self._range_from_48h_arrays(
            self._tariff_48_seconds,
            self._tariff_48_values,
            start_second=start_second,
            period=period,
            step_seconds=step_seconds,
            default=0.0,
        )

        if self.type == 1:
            feed_vals = tariff_vals.copy() if copy else tariff_vals
        else:
            feed_vals = self._range_from_48h_arrays(
                self._feed_48_seconds,
                self._feed_48_values,
                start_second=start_second,
                period=period,
                step_seconds=step_seconds,
                default=0.0,
            )

        if copy:
            return tariff_vals.copy(), feed_vals.copy()

        return tariff_vals, feed_vals

    def get_tariff_range_df(
        self,
        now_time: datetime,
        period: int = 96,
        resolution: timedelta = timedelta(minutes=15),
    ) -> pd.DataFrame:
        """
        Backward-compatible DataFrame API.

        Internally this now uses fast NumPy lookup, then wraps the result in a DataFrame.
        Use get_tariff_range_array() in performance-critical code.
        """
        tariff_vals, feed_vals = self.get_tariff_range_array(
            now_time=now_time,
            period=period,
            resolution=resolution,
            copy=False,
        )

        idx = pd.date_range(
            start=pd.Timestamp(now_time),
            periods=period,
            freq=pd.Timedelta(resolution),
        )

        df = pd.DataFrame(
            {
                "tariff": tariff_vals.astype(float, copy=False),
                "feed_tariff": feed_vals.astype(float, copy=False),
            },
            index=idx,
        )
        df.index.name = "time"
        return df

    # -------------------------------------------------------------------------
    # Commit staged tariffs
    # -------------------------------------------------------------------------
    def updated_tariff(self) -> None:
        """
        If the next-day tariffs are staged, commit them and refresh min/max/caches.
        """
        if self.next_24hr_tariff is not None:
            logger.commandline("Updating tariff value")
            self.tariff = self.next_24hr_tariff
            self.next_24hr_tariff = None

        if self.next_24hr_feed_tariff is not None:
            logger.commandline("Updating feed tariff value")
            self.feed_tariff = self.next_24hr_feed_tariff
            self.next_24hr_feed_tariff = None

        if self.type == 1 and self.tariff is not None:
            self.feed_tariff = self.tariff.copy()

        self._next_tariff_seconds = None
        self._next_tariff_values = None
        self._next_feed_seconds = None
        self._next_feed_values = None

        self._sync_current_arrays()
        self.update_min_max()

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def update_min_max(self) -> None:
        """
        Update min/max/average stats from NumPy arrays where possible.
        """
        if self._tariff_values is None:
            self._tariff_seconds, self._tariff_values = self._df_to_arrays(self.tariff)

        if self._feed_values is None:
            if self.type == 1:
                self._feed_seconds = self._tariff_seconds
                self._feed_values = self._tariff_values
            else:
                self._feed_seconds, self._feed_values = self._df_to_arrays(self.feed_tariff)

        if self._tariff_values is not None and len(self._tariff_values) > 0:
            self.max_tariff = float(np.max(self._tariff_values))
            self.min_tariff = float(np.min(self._tariff_values))
            self.avg_tariff = float(np.mean(self._tariff_values))

        if self.type == 1:
            self.max_feed_tariff = self.max_tariff
            self.min_feed_tariff = self.min_tariff
            self.avg_feed_tariff = self.avg_tariff
        elif self._feed_values is not None and len(self._feed_values) > 0:
            self.max_feed_tariff = float(np.max(self._feed_values))
            self.min_feed_tariff = float(np.min(self._feed_values))
            self.avg_feed_tariff = float(np.mean(self._feed_values))

    # -------------------------------------------------------------------------
    # Convenience/debug helpers
    # -------------------------------------------------------------------------
    def get_current_tariff_arrays(self, copy: bool = False) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return current-day import tariff arrays:
            seconds_of_day, values
        """
        if copy:
            s = None if self._tariff_seconds is None else self._tariff_seconds.copy()
            v = None if self._tariff_values is None else self._tariff_values.copy()
            return s, v
        return self._tariff_seconds, self._tariff_values

    def get_current_feed_tariff_arrays(self, copy: bool = False) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return current-day feed tariff arrays:
            seconds_of_day, values
        """
        if copy:
            s = None if self._feed_seconds is None else self._feed_seconds.copy()
            v = None if self._feed_values is None else self._feed_values.copy()
            return s, v
        return self._feed_seconds, self._feed_values
