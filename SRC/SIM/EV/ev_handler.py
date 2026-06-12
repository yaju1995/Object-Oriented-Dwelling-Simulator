from __future__ import annotations

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from typing import Optional

from SRC.SIM.ESS.ess_handler import ESSHandler
from SRC.support import CustomLogger
from .ev_profile_generator import generate_ev_sessions


logger = CustomLogger(command=False)


class EVHandler(ESSHandler):
    """
    EV model driven by a CSV profile.

    Optimized version:
    - Uses NumPy arrays for EV sessions.
    - Uses np.searchsorted() for fast session lookup.
    - Avoids pandas filtering during every simulation step.
    """

    def __init__(
        self,
        name: str,
        ev_profile_csv: str,
        start_time: datetime,
        resolution: timedelta,
        duration: timedelta,
        total_capacity_Wh: float,
        charging_power_W: float,
        discharging_power_W: float,
        upper_limit_soc_pct: float,
        lower_limit_soc_pct: float,
        v2g_enabled: bool = False,
        in_eff: float = 1.0,
        out_eff: float = 1.0,
        seed: int = 42,
    ):
        # -------------------------------------------------
        # Load EV profile
        # -------------------------------------------------
        self.ev_df = generate_ev_sessions(
            ev_profile_csv,
            start_time,
            resolution,
            duration,
            seed=seed,
        )

        self._validate_ev_profile()
        self._compile_ev_session_arrays()

        # -------------------------------------------------
        # Initialize ESS
        # -------------------------------------------------
        super().__init__(
            name=name,
            total_capacity_Wh=total_capacity_Wh,
            initial_soc_pct=0.0,
            charging_power_W=charging_power_W,
            discharging_power_W=discharging_power_W,
            resolution=resolution,
            upper_limit_soc_pct=upper_limit_soc_pct,
            lower_limit_soc_pct=lower_limit_soc_pct,
            in_eff=in_eff,
            out_eff=out_eff,
        )

        self.v2g_enabled = v2g_enabled

        # Runtime state
        self.active_session_idx: Optional[int] = None
        self.previous_plugged: bool = False

        self.user_set_plugout = None
        self.expected_soc = 100

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_ev_profile(self):
        required = {
            "plug_in_time",
            "plug_out_time",
            "initial_soc",
            "day_id",
            "weekday",
        }

        missing = required - set(self.ev_df.columns)
        if missing:
            raise ValueError(f"EV profile missing columns: {missing}")

        for col in ["plug_in_time", "plug_out_time"]:
            series = self.ev_df[col]

            if not pd.api.types.is_datetime64_any_dtype(series):
                if not series.isna().all():
                    raise TypeError(f"{col} must be datetime or None")

        valid_rows = (
            self.ev_df["plug_in_time"].notna()
            & self.ev_df["plug_out_time"].notna()
        )

        if (
            self.ev_df.loc[valid_rows, "plug_out_time"]
            <= self.ev_df.loc[valid_rows, "plug_in_time"]
        ).any():
            raise ValueError("plug_out_time must be after plug_in_time")

    # ==================================================================
    # Compile NumPy arrays
    # ==================================================================

    def _compile_ev_session_arrays(self):
        """
        Precompute NumPy arrays for fast runtime lookup.
        """

        self.ev_df = self.ev_df.dropna(
            subset=["plug_in_time", "plug_out_time"]
        ).copy()

        self.ev_df = self.ev_df.sort_values("plug_in_time").reset_index(drop=True)

        self.plug_in_arr = self.ev_df["plug_in_time"].to_numpy(dtype="datetime64[ns]")
        self.plug_out_arr = self.ev_df["plug_out_time"].to_numpy(dtype="datetime64[ns]")

        self.initial_soc_arr = self.ev_df["initial_soc"].to_numpy(dtype=np.float64)
        self.day_id_arr = self.ev_df["day_id"].to_numpy()
        self.weekday_arr = self.ev_df["weekday"].to_numpy()

        self.n_sessions = len(self.ev_df)

    # ==================================================================
    # Session lookup
    # ==================================================================

    def _find_active_session(self, timestamp: datetime) -> Optional[int]:
        """
        Fast active EV session lookup.

        Returns
        -------
        int or None
            Index of active EV session, or None if EV is not plugged in.
        """

        if self.n_sessions == 0:
            return None

        ts = np.datetime64(timestamp, "ns")

        # Fast path: current session still active
        if self.active_session_idx is not None:
            idx = self.active_session_idx

            if self.plug_in_arr[idx] <= ts < self.plug_out_arr[idx]:
                return idx

        # Binary search for latest plug-in time <= timestamp
        idx = np.searchsorted(self.plug_in_arr, ts, side="right") - 1

        if idx < 0:
            return None

        if ts < self.plug_out_arr[idx]:
            return int(idx)

        return None

    # ==================================================================
    # Step
    # ==================================================================

    def step(
        self,
        timestamp: datetime,
        control_power_W: Optional[float] = 0,
    ) -> dict|tuple:
        """
        Advance EV by one timestep.
        """

        session_idx = self._find_active_session(timestamp)
        plugged = session_idx is not None

        # -------------------------------------------------
        # Plug-in event
        # -------------------------------------------------
        if plugged and not self.previous_plugged:
            init_soc = float(self.initial_soc_arr[session_idx])

            self.user_set_plugout = pd.Timestamp(
                self.plug_out_arr[session_idx]
            ).to_pydatetime()

            self.expected_soc = 100

            logger.commandline(
                f"[EV] Plug-in @ {timestamp} | "
                f"SOC reset → {init_soc:.2f}% | "
                f"day_id={self.day_id_arr[session_idx]} "
                f"weekday={self.weekday_arr[session_idx]} | "
                f"Plug-out @ {self.user_set_plugout}"
            )

            self.set_soc(init_soc, normalized=False, clip=False)
            self.active_session_idx = session_idx

        # -------------------------------------------------
        # Plug-out event
        # -------------------------------------------------
        if not plugged and self.previous_plugged:
            logger.commandline(
                f"[EV] Plug-out @ {timestamp} | "
                f"SOC={self.getStateOfCharge():.2f}%"
            )

            self.active_session_idx = None

        self.previous_plugged = plugged

        # -------------------------------------------------
        # Power decision
        # -------------------------------------------------
        if not plugged:
            power_setpoint_W = 0.0
            self.user_set_plugout = None
        else:
            power_setpoint_W = 0.0 if control_power_W is None else float(control_power_W)

            if not self.v2g_enabled:
                power_setpoint_W = max(0.0, power_setpoint_W)

        # -------------------------------------------------
        # ESS update
        # -------------------------------------------------
        ev_soc, ev_power_kw, ev_set_power_kw = self.update(
            power_setpoint_W=power_setpoint_W,
            timestamp=timestamp,
        )
        ev_soc =  ev_soc if plugged else 0.0
        ################################################################
        # ev_power_kw = out["Battery Electric Power (kW)"]
        # ev_set_power_kw = out["Battery Set Power (kW)"]
        # ev_soc = out["Battery SOC (-)"] if plugged else 0.0

        # return {
        #     "EV Electric Power (kW)": ev_power_kw,
        #     "EV Set Power (kW)": ev_set_power_kw,
        #     "EV SOC (-)": ev_soc,
        #     "EV Parked": 1 if plugged else 0,
        #     "User Plug Out Time": self.user_set_plugout,
        #     "Expected SOC": self.expected_soc,
        # }

        return (
            ev_power_kw,
            ev_set_power_kw,
            ev_soc,
            1 if plugged else 0,
            self.user_set_plugout,
            self.expected_soc,
        )

    # ==================================================================
    # EV SOC
    # ==================================================================

    def getEVStateOfCharge(self, timestamp: datetime) -> float:
        return self.getStateOfCharge() if self._find_active_session(timestamp) else 0.0

    # ==================================================================
    # Upload EV profile
    # ==================================================================

    def upload_ev_df_from_csv(self, csv_path: str):
        """
        Directly load EV session DataFrame from CSV.

        Expected columns:
            plug_in_time, plug_out_time, initial_soc, day_id, weekday
        """

        logger.commandline(f"[EV] Loading EV profile from CSV: {csv_path}")

        df = pd.read_csv(
            csv_path,
            parse_dates=["plug_in_time", "plug_out_time"],
        )

        df = df.sort_values("plug_in_time").reset_index(drop=True)

        self.ev_df = df
        self._validate_ev_profile()
        self._compile_ev_session_arrays()

        self.active_session_idx = None
        self.previous_plugged = False
        self.user_set_plugout = None
        self.expected_soc = 100

        logger.commandline(
            f"[EV] Loaded {len(self.ev_df)} EV sessions from CSV"
        )