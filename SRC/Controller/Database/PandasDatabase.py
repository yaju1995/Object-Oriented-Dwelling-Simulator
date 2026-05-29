from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
import pandas as pd


@dataclass
class DataStore:
    resolution: timedelta                    # ✅ base resolution as datetime.timedelta
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    lock: RLock = field(default_factory=RLock)

    # -------------------- write --------------------
    def append(self, t: datetime | pd.Timestamp, row: dict) -> None:
        t = pd.Timestamp(t)
        with self.lock:
            if self.df.empty:
                self.df = pd.DataFrame([row], index=pd.DatetimeIndex([t], name="time"))
            else:
                self.df.loc[t] = row
            self.df.sort_index(inplace=True)


    def get_instant_state(
            self,
            keys: list[str],
            *,
            now_time: datetime | pd.Timestamp | None = None,
            strict: bool = True,
    ) -> pd.Series | None:
        """
        Return the latest values for selected keys.

        If now_time is provided, returns the latest row <= now_time.
        If strict=True, missing keys raise an error.
        """

        with self.lock:
            if self.df.empty:
                return None

            df = self.df

            # ----- choose row -----
            if now_time is None:
                row = df.iloc[-1]
            else:
                now_time = pd.Timestamp(now_time)
                pos = df.index.get_indexer([now_time], method="pad")[0]
                if pos == -1:
                    return None
                row = df.iloc[pos]

            # ----- key validation -----
            missing = set(keys) - set(df.columns)
            if missing:
                if strict:
                    raise KeyError(f"Missing instant keys: {missing}")
                else:
                    return row.reindex(keys)

            return row[keys].copy()

