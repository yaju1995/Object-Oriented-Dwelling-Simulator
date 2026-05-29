"""
Simulator_fast.py

A faster version of the dwelling simulator.

Main design change:
- pandas is used for setup, validation, upload, and final export.
- NumPy arrays are used inside the simulation loop.

This avoids expensive per-step DataFrame operations such as:
    self.simulation_df.loc[t, col] = value
    self.simulation_df.reset_index().loc[...]

Expected usage:
    sim = DwellingFast(...)
    sim.initialized_df()
    obs = sim.step(control_signal)
    results = sim.to_dataframe()

This file keeps compatibility with your existing handler classes:
    ESSHandler
    EVHandler
    ThermalHandler
    EPWWeatherHandler
    tariffHandler
    PatternGenerationHandler
    EquipmentClass models
"""

from __future__ import annotations

import datetime
import datetime as dt
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from SRC.SIM.ESS.ess_handler import ESSHandler
from SRC.SIM.EV.ev_handler import EVHandler
from SRC.SIM.Thermal.thermal_handler import ThermalHandler
from SRC.SIM.Weather.epwhandler_V_1_0_3 import EPWWeatherHandler
# from SRC.SIM.Tariff.tariffHandler_V_2 import tariffHandler
from SRC.SIM.Tariff.tariffHandler_V_2_numpy import tariffHandler
from SRC.SIM.DataGenerator.data_generators import PatternGenerationHandler
from SRC.support.lib_config import CustomLogger

from SRC.SIM.EquipmentClass import InverterModel, EVModel, MeterModel, HVACModel

np.set_printoptions(suppress=True, precision=2)

logger = CustomLogger(command=True)

# Information collection keys
NET_POWER = ["Total Electric Power (kW)", "Total Reactive Power (kVAR)"]
INVERTER = ["PV Electric Power (kW)", "Battery Electric Power (kW)", "Battery SOC (-)"]
EV = ["EV Parked", "EV SOC (-)", "EV Electric Power (kW)", "User Plug Out Time", "Expected SOC"]
HVAC = ["Temperature - Indoor (C)", "HVAC Heating Electric Power (kW)", "Temperature - Outdoor (C)"]


class SimulationEnded(Exception):
    """Raised when the simulation reaches the final timestep."""


class DwellingFast:
    """
    Fast dwelling simulator.

    Compatibility notes:
    - The class name is capitalised. If your existing code expects `dwelling`, an alias is
      provided at the bottom of the file: `dwelling = DwellingFast`.
    - `simulation_df` is still created during initialisation and can still be exported.
    - The runtime loop uses NumPy arrays rather than mutating `simulation_df` each step.
    """

    def __init__(
        self,
        name: str,
        start_time: dt.datetime,
        duration: dt.timedelta,
        resolution: dt.timedelta,
        demand_config: dict | None = None,
        weather_file: str | None = None,
        pv_config: dict | None = None,
        battery_config: dict | None = None,
        ev_config: dict | None = None,
        thermal_config: dict | None = None,
        PV: float = 0,
        tariff: tariffHandler | None = None,
        seed: int = 42,
    ):
        self.name = name
        self.start_time = start_time
        self.duration = duration
        self.resolution = resolution
        self.end_time = self.start_time + self.duration
        self.now_time: dt.datetime | pd.Timestamp | None = None

        logger.commandline(f"Start time: {self.start_time}: End time:{self.end_time}")

        self.weather_file = weather_file
        self.pv_config = pv_config
        self.demand_config = demand_config
        self.battery_config = battery_config
        self.ev_config = ev_config
        self.thermal_config = thermal_config

        self.Battery: ESSHandler | None = None
        self.EV: EVHandler | None = None
        self.Thermal: ThermalHandler | None = None

        # Important: use inclusive='left' so duration has exactly duration/resolution steps.
        # If you want the old behaviour with start and end both included, set inclusive='both'.
        self.time_index = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=self.resolution,
            inclusive="left",
        )
        self.simulation_df = pd.DataFrame(index=self.time_index)
        self.simulation_df.index.name = "Time"

        self.tariff = tariff if tariff is not None else tariffHandler()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # IoT device model outputs
        self.MeterModel = MeterModel()
        self.InverterModel = InverterModel()
        self.EVModel = EVModel()
        self.HVACModel = HVACModel()

        # Runtime state
        self.compiled = False
        self.n_steps = len(self.time_index)
        self.step_idx = -1

    # ---------------------------------------------------------------------
    # Setup / initialisation
    # ---------------------------------------------------------------------
    def initialized_df(self) -> None:
        """
        Prepare input/output columns and initialise handlers.

        This still uses pandas because this stage is not usually the bottleneck.
        The fast runtime arrays are compiled at the end.
        """

        # Meter defaults
        meter_df = pd.DataFrame(
            0.0,
            index=self.simulation_df.index,
            columns=["Total Electric Power (kW)", "Total Reactive Power (kVAR)"],
        )
        self.simulation_df = self.simulation_df.join(meter_df)

        # ---------------- Demand ----------------
        if self.demand_config:
            logger.commandline("Setting up demand profile for simulation")
            if not isinstance(self.demand_config, dict):
                raise ValueError("demand_config must be a dictionary")

            required_keys = {"model", "file"}
            missing = required_keys - self.demand_config.keys()
            if missing:
                raise ValueError(f"Missing required keys in demand_config: {missing}")

            model = self.demand_config.get("model")
            if model == "default":
                model = "normal"
                file_path = "./SRC/SIM/Defaults/Demand/15_min_normal_test.csv"
            else:
                file_path = self.demand_config.get("file")

            demand_handler = PatternGenerationHandler(
                model_name=model,
                csv_path=file_path,
            )
            demand_df = demand_handler.get_simulation_data(
                start_date=self.start_time,
                duration_days=self.duration,
                resolution=self.resolution,
                seed=self.seed + 1,
                column_name="Demand Electric Power (kW)",
            )
            demand_df = self._align_to_sim_index(demand_df)
        else:
            logger.commandline("No demand model detected, setting demand to 0")
            demand_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=["Demand Electric Power (kW)"],
            )

        self.simulation_df = self._safe_join_replace(self.simulation_df, demand_df)

        # ---------------- Weather ----------------
        if self.weather_file:
            logger.commandline("Weather file detected, setting up weather information")
            weather_handler = EPWWeatherHandler(self.weather_file)
            weather_df = weather_handler.get_simulation_data(
                start_date=self.start_time,
                resolution=self.resolution,
                duration=self.duration,
            )
            weather_df = self._align_to_sim_index(weather_df)
            self.simulation_df = self._safe_join_replace(self.simulation_df, weather_df)
        else:
            logger.commandline("No weather file detected")

        # ---------------- PV ----------------
        if self.pv_config:
            logger.commandline(self.pv_config)
            required_keys = {"type", "capacity W", "efficiency", "area per W", "tilt", "azimuth"}
            missing = required_keys - self.pv_config.keys()
            if missing:
                raise ValueError(f"Missing required keys in pv_config: {missing}")

            pv_rating_kw = self.pv_config.get("capacity W", 0) / 1000.0
            pv_efficiency = self.pv_config.get("efficiency", 1.0)
            pv_config_type = self.pv_config.get("type")

            if pv_config_type == "train":
                logger.commandline("PV training config detected: generating random irradiance")
                self.simulation_df["Global Horizontal Radiation"] = self.rng.uniform(
                    low=0,
                    high=self.pv_config.get("max_irradiance", 1000),
                    size=len(self.simulation_df),
                )
            else:
                logger.commandline("PV config detected: using weather file irradiance")

            if "Global Horizontal Radiation" not in self.simulation_df.columns:
                self.simulation_df["Global Horizontal Radiation"] = 0.0

            ghi = self.simulation_df["Global Horizontal Radiation"].fillna(0.0)
            pv_series = np.round((ghi / 1000.0) * pv_efficiency * pv_rating_kw, 3)
            pv_series = pv_series.clip(lower=0)
            pv_series.name = "PV Electric Power (kW)"
            self.simulation_df = self._safe_join_replace(self.simulation_df, pv_series.to_frame())
        else:
            self.simulation_df["PV Electric Power (kW)"] = 0.0

        # ---------------- EV ----------------
        if self.ev_config or self.EV:
            logger.commandline("==== Initializing EV ====")
            ev_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=[
                    "EV Electric Power (kW)",
                    "EV Set Power (kW)",
                    "EV SOC (-)",
                    "EV Parked",
                    "Expected SOC",
                ],
            )
            ev_df["User Plug Out Time"] = None
            self.simulation_df = self._safe_join_replace(self.simulation_df, ev_df)

            if self.ev_config:
                required_keys = {
                    "capacity Wh",
                    "charging power W",
                    "discharging power W",
                    "charging eff",
                    "discharging eff",
                    "v2g_enabled",
                    "profile_file",
                }
                missing = required_keys - self.ev_config.keys()
                if missing:
                    raise ValueError(f"Missing required keys in ev_config: {missing}")

                discharging_power_w = self.ev_config.get("discharging power W")
                if not self.ev_config.get("v2g_enabled"):
                    discharging_power_w = 0

                self.EV = EVHandler(
                    name="EV_1",
                    ev_profile_csv=self.ev_config.get("profile_file"),
                    start_time=self.start_time,
                    resolution=self.resolution,
                    duration=self.duration,
                    total_capacity_Wh=self.ev_config.get("capacity Wh"),
                    charging_power_W=self.ev_config.get("charging power W"),
                    discharging_power_W=discharging_power_w,
                    v2g_enabled=self.ev_config.get("v2g_enabled"),
                    upper_limit_soc_pct=100,
                    lower_limit_soc_pct=20,
                    seed=self.seed,
                    in_eff=self.ev_config.get("charging eff"),
                    out_eff=self.ev_config.get("discharging eff"),
                )
            elif not isinstance(self.EV, EVHandler):
                logger.commandline("EV must be EVHandler; ignoring EV setup")
        else:
            logger.commandline("No EV found: ignoring EV setup")

        # ---------------- ESS ----------------
        if self.battery_config or self.Battery:
            logger.commandline("==== Initializing ESS ====")
            ess_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=[
                    "Battery Electric Power (kW)",
                    "Battery Set Power (kW)",
                    "Battery SOC (-)",
                ],
            )
            self.simulation_df = self._safe_join_replace(self.simulation_df, ess_df)

            if self.battery_config:
                required_keys = {
                    "capacity Wh",
                    "initial soc",
                    "charging power W",
                    "discharging power W",
                    "charging eff",
                    "discharging eff",
                }
                missing = required_keys - self.battery_config.keys()
                if missing:
                    raise ValueError(f"Missing required keys in battery_config: {missing}")

                self.Battery = ESSHandler(
                    name=f"HouseESS_{self.name}",
                    total_capacity_Wh=self.battery_config.get("capacity Wh"),
                    initial_soc_pct=self.battery_config.get("initial soc"),
                    charging_power_W=self.battery_config.get("charging power W"),
                    discharging_power_W=self.battery_config.get("discharging power W"),
                    resolution=self.resolution,
                    in_eff=self.battery_config.get("charging eff"),
                    out_eff=self.battery_config.get("discharging eff"),
                )
            elif not isinstance(self.Battery, ESSHandler):
                logger.commandline("Battery must be ESSHandler; ignoring ESS setup")
        else:
            logger.commandline("No ESS found: ignoring ESS setup")

        # ---------------- Thermal ----------------
        if self.thermal_config or self.Thermal:
            logger.commandline("==== Initializing Thermal Loads ====")
            thermal_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=[
                    "HVAC Heating Electric Power (kW)",
                    "HVAC Heating Setpoint (C)",
                    "Temperature - Indoor (C)",
                ],
            )
            self.simulation_df = self._safe_join_replace(self.simulation_df, thermal_df)

            if self.thermal_config:
                required_keys = {"initial temperature C", "tau", "W", "n"}
                missing = required_keys - self.thermal_config.keys()
                if missing:
                    raise ValueError(f"Missing required keys in thermal_config: {missing}")

                self.Thermal = ThermalHandler(
                    resolution=self.resolution,
                    initial_internal_temperature=self.thermal_config.get("initial temperature C"),
                    tau=self.thermal_config.get("tau"),
                    W=self.thermal_config.get("W"),
                    n=self.thermal_config.get("n"),
                )
            elif not isinstance(self.Thermal, ThermalHandler):
                logger.commandline("Thermal model must be ThermalHandler; ignoring thermal setup")
        else:
            logger.commandline("No thermal load detected")

        self.compile_runtime_arrays()
        self.compile_tariff_arrays()

    @staticmethod
    def _safe_join_replace(base: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        """Join incoming columns, replacing same-name columns first."""
        overlap = [c for c in incoming.columns if c in base.columns]
        if overlap:
            base = base.drop(columns=overlap)
        return base.join(incoming)

    def _align_to_sim_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align dataframe to the simulator time index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Time" in df.columns:
                df = df.set_index("Time")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                raise ValueError("DataFrame must have a DatetimeIndex, Time column, or timestamp column")

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Clip/reindex exactly to simulator index.
        df = df.reindex(self.simulation_df.index)
        return df

    # ---------------------------------------------------------------------
    # Upload / validation
    # ---------------------------------------------------------------------
    def upload_data(self, file: str, columns: list[str] | str | None = None) -> None:
        """
        Upload CSV data and update selected simulation columns.

        Requirements:
        - CSV must contain a 'timestamp' column.
        - Data is clipped/reindexed to the simulation index.
        - After upload, runtime arrays are recompiled.
        """
        logger.commandline("Uploading data for simulation")

        df = pd.read_csv(file)
        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain a 'timestamp' column.")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        if columns is None:
            columns = list(df.columns)
        elif isinstance(columns, str):
            columns = [columns]

        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        df = df[columns]
        df = self._align_to_sim_index(df)

        if df.isna().any().any():
            raise ValueError("Uploaded data does not fully cover the simulation time index.")

        self.simulation_df = self._safe_join_replace(self.simulation_df, df)
        self.compile_runtime_arrays()
        self.compile_tariff_arrays()

        logger.commandline(f"Updated columns: {sorted(columns)}")

    def upload_time_series(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        agg: str = "mean",
    ) -> pd.DataFrame:
        """
        Validate, clip, and resample dataframe based on class time settings.
        """
        df = df.copy()

        if timestamp_col not in df.columns:
            raise ValueError(f"Missing timestamp column: {timestamp_col}")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).set_index(timestamp_col)
        df = df.loc[self.start_time:self.end_time]

        if df.empty:
            raise ValueError("No data available in simulation window.")

        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            raise ValueError("Cannot infer resolution.")

        current_resolution = diffs.mode().iloc[0]
        if current_resolution != self.resolution:
            rule = f"{int(self.resolution.total_seconds() // 60)}min"
            if agg == "mean":
                df = df.resample(rule).mean()
            elif agg == "sum":
                df = df.resample(rule).sum()
            elif agg == "first":
                df = df.resample(rule).first()
            else:
                raise ValueError("agg must be 'mean', 'sum', or 'first'")

        return df.reindex(self.simulation_df.index)

    # ---------------------------------------------------------------------
    # Runtime array compilation
    # ---------------------------------------------------------------------
    def compile_runtime_arrays(self) -> None:
        """
        Convert simulation_df columns into NumPy arrays for fast runtime access.
        Call after setup or after uploading external data.
        """
        df = self.simulation_df
        self.time_index = df.index
        self.n_steps = len(df)
        self.step_idx = -1
        self.now_time = None

        def col_array(name: str, default: float = 0.0, dtype: Any = np.float64) -> np.ndarray:
            if name in df.columns:
                return df[name].fillna(default).to_numpy(dtype=dtype, copy=True)
            return np.full(self.n_steps, default, dtype=dtype)

        # Inputs
        self.demand_kw_arr = col_array("Demand Electric Power (kW)")
        self.pv_kw_arr = col_array("PV Electric Power (kW)")
        self.outdoor_temp_arr = col_array("Temperature - Outdoor (C)", default=np.nan)

        # Meter outputs
        self.total_kw_arr = col_array("Total Electric Power (kW)")
        self.total_kvar_arr = col_array("Total Reactive Power (kVAR)")

        # EV outputs
        self.ev_kw_arr = col_array("EV Electric Power (kW)")
        self.ev_set_kw_arr = col_array("EV Set Power (kW)")
        self.ev_soc_arr = col_array("EV SOC (-)")
        self.ev_parked_arr = col_array("EV Parked")
        self.ev_expected_soc_arr = col_array("Expected SOC")
        self.ev_plug_out_time_arr = np.empty(self.n_steps, dtype=object)

        if "User Plug Out Time" in df.columns:
            values = df["User Plug Out Time"].to_numpy(dtype=object, copy=True)
            self.ev_plug_out_time_arr[:] = values
        else:
            self.ev_plug_out_time_arr[:] = None

        # ESS outputs
        self.battery_kw_arr = col_array("Battery Electric Power (kW)")
        self.battery_set_kw_arr = col_array("Battery Set Power (kW)")
        self.battery_soc_arr = col_array("Battery SOC (-)")

        # Thermal outputs
        self.hvac_kw_arr = col_array("HVAC Heating Electric Power (kW)")
        self.indoor_temp_arr = col_array("Temperature - Indoor (C)")

        self.compiled = True

    def compile_tariff_arrays(self) -> None:
        """
        Precompute tariff and feed-in tariff for each simulation step.
        This avoids repeated tariff calculation inside the runtime loop.
        """
        self.tariff_arr = np.zeros(self.n_steps, dtype=np.float64)
        self.feed_tariff_arr = np.zeros(self.n_steps, dtype=np.float64)

        if self.tariff is None:
            return

        for i, t in enumerate(self.time_index):
            tariff_value, feed_tariff_value = self.tariff.get_tariff(t.time())
            self.tariff_arr[i] = float(tariff_value)
            self.feed_tariff_arr[i] = float(feed_tariff_value)

    # ---------------------------------------------------------------------
    # Fast stepping
    # ---------------------------------------------------------------------
    def _advance_step(self) -> tuple[int, pd.Timestamp]:
        if not self.compiled:
            raise RuntimeError("Simulator is not compiled. Call initialized_df() first.")

        self.step_idx += 1
        if self.step_idx >= self.n_steps:
            raise StopIteration("Simulation finished")

        self.now_time = self.time_index[self.step_idx]
        return self.step_idx, self.now_time

    def get_next_24h_tariffs_fast(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return next-24h tariff arrays by slicing precomputed tariff arrays.
        If the simulation is near the end, the returned arrays are shorter.
        """
        steps_per_hour = max(1, int(dt.timedelta(hours=1) / self.resolution))
        window_steps = 24 * steps_per_hour
        end = min(i + window_steps, self.n_steps)
        return self.tariff_arr[i:end], self.feed_tariff_arr[i:end]

    def update(self, control: dict | None) -> dict:
        """
        Fast update method.

        Uses NumPy arrays rather than pandas row/cell operations.
        """
        if control is None:
            control = {}

        i, t = self._advance_step()

        demand_kw = float(self.demand_kw_arr[i])
        pv_kw = float(self.pv_kw_arr[i])

        # ---------------- EV ----------------
        ev_kw = 0.0
        ev_set_kw = 0.0
        ev_soc = 0.0
        ev_parked = 0.0
        user_plug_out_time = None
        expected_soc = 0.0

        if self.EV:
            ev_control = control.get("EV", {})
            ev_power_set = ev_control.get("Max Power", 0)

            ev_status = self.EV.step(
                control_power_W=ev_power_set,
                timestamp=t,
            )

            ev_kw = float(ev_status.get("EV Electric Power (kW)", 0.0) or 0.0)
            ev_set_kw = float(ev_status.get("EV Set Power (kW)", 0.0) or 0.0)
            ev_soc = float(ev_status.get("EV SOC (-)", 0.0) or 0.0)
            ev_parked = float(ev_status.get("EV Parked", 0.0) or 0.0)
            user_plug_out_time = ev_status.get("User Plug Out Time")
            expected_soc = float(ev_status.get("Expected SOC", 0.0) or 0.0)

            self.ev_kw_arr[i] = ev_kw
            self.ev_set_kw_arr[i] = ev_set_kw
            self.ev_soc_arr[i] = ev_soc
            self.ev_parked_arr[i] = ev_parked
            self.ev_plug_out_time_arr[i] = user_plug_out_time
            self.ev_expected_soc_arr[i] = expected_soc

        # ---------------- Thermal ----------------
        thermal_kw = 0.0
        indoor_temp = float(self.indoor_temp_arr[i]) if i < len(self.indoor_temp_arr) else 0.0

        if self.Thermal:
            hvac_control = control.get("HVAC Heating", {})
            power_p_set = hvac_control.get("P Setpoint", 0)

            thermal_kw = abs(float(power_p_set) / 1000.0)
            outdoor_temp = self.outdoor_temp_arr[i]

            indoor_temp = float(
                self.Thermal.update(
                    power_W=control.get("thermal_kw", power_p_set),
                    external_temperature=float(outdoor_temp),
                )
            )

            self.hvac_kw_arr[i] = thermal_kw
            self.indoor_temp_arr[i] = indoor_temp

        # ---------------- Battery ----------------
        battery_kw = 0.0
        battery_set_kw = 0.0
        battery_soc = 0.0

        if self.Battery:
            battery_control = control.get("Battery", {})
            power_p_set = battery_control.get("P Setpoint", 0)

            battery_status = self.Battery.update(
                power_setpoint_W=power_p_set,
                timestamp=t,
            )

            battery_kw = float(battery_status.get("Battery Electric Power (kW)", 0.0) or 0.0)
            battery_set_kw = float(battery_status.get("Battery Set Power (kW)", 0.0) or 0.0)
            battery_soc = float(battery_status.get("Battery SOC (-)", 0.0) or 0.0)

            self.battery_kw_arr[i] = battery_kw
            self.battery_set_kw_arr[i] = battery_set_kw
            self.battery_soc_arr[i] = battery_soc

        # ---------------- Power balance ----------------
        total_load_kw = demand_kw + ev_kw + thermal_kw
        total_generation_kw = pv_kw - battery_kw
        net_active_kw = round(total_load_kw - total_generation_kw, 3)

        self.total_kw_arr[i] = net_active_kw
        self.total_kvar_arr[i] = net_active_kw

        return {
            "Time": t,
            "Demand Electric Power (kW)": demand_kw,
            "PV Electric Power (kW)": pv_kw,
            "EV Electric Power (kW)": ev_kw,
            "EV Set Power (kW)": ev_set_kw,
            "EV SOC (-)": ev_soc,
            "EV Parked": ev_parked,
            "User Plug Out Time": user_plug_out_time,
            "Expected SOC": expected_soc,
            "Battery Electric Power (kW)": battery_kw,
            "Battery Set Power (kW)": battery_set_kw,
            "Battery SOC (-)": battery_soc,
            "HVAC Heating Electric Power (kW)": thermal_kw,
            "Temperature - Indoor (C)": indoor_temp,
            "Temperature - Outdoor (C)": float(self.outdoor_temp_arr[i]) if not np.isnan(self.outdoor_temp_arr[i]) else np.nan,
            "Total Electric Power (kW)": net_active_kw,
            "Total Reactive Power (kVAR)": net_active_kw,
        }

    from time import perf_counter

    def update_profiled(self, control: dict | None) -> dict:
        if control is None:
            control = {}

        t0 = perf_counter()

        i, t = self._advance_step()

        demand_kw = float(self.demand_kw_arr[i])
        pv_kw = float(self.pv_kw_arr[i])

        t_base = perf_counter()

        # ---------------- EV ----------------
        ev_kw = 0.0
        ev_set_kw = 0.0
        ev_soc = 0.0
        ev_parked = 0.0
        user_plug_out_time = None
        expected_soc = 0.0

        if self.EV:
            ev_control = control.get("EV", {})
            ev_power_set = ev_control.get("Max Power", 0)

            ev_status = self.EV.step(
                control_power_W=ev_power_set,
                timestamp=t,
            )

            ev_kw = float(ev_status.get("EV Electric Power (kW)", 0.0) or 0.0)
            ev_set_kw = float(ev_status.get("EV Set Power (kW)", 0.0) or 0.0)
            ev_soc = float(ev_status.get("EV SOC (-)", 0.0) or 0.0)
            ev_parked = float(ev_status.get("EV Parked", 0.0) or 0.0)
            user_plug_out_time = ev_status.get("User Plug Out Time")
            expected_soc = float(ev_status.get("Expected SOC", 0.0) or 0.0)

            self.ev_kw_arr[i] = ev_kw
            self.ev_set_kw_arr[i] = ev_set_kw
            self.ev_soc_arr[i] = ev_soc
            self.ev_parked_arr[i] = ev_parked
            self.ev_plug_out_time_arr[i] = user_plug_out_time
            self.ev_expected_soc_arr[i] = expected_soc

        t_ev = perf_counter()

        # ---------------- Thermal ----------------
        thermal_kw = 0.0
        indoor_temp = float(self.indoor_temp_arr[i])

        if self.Thermal:
            hvac_control = control.get("HVAC Heating", {})
            power_p_set = hvac_control.get("P Setpoint", 0)

            thermal_kw = abs(float(power_p_set) / 1000.0)
            outdoor_temp = self.outdoor_temp_arr[i]

            indoor_temp = float(
                self.Thermal.update(
                    power_W=control.get("thermal_kw", power_p_set),
                    external_temperature=float(outdoor_temp),
                )
            )

            self.hvac_kw_arr[i] = thermal_kw
            self.indoor_temp_arr[i] = indoor_temp

        t_thermal = perf_counter()

        # ---------------- Battery ----------------
        battery_kw = 0.0
        battery_set_kw = 0.0
        battery_soc = 0.0

        if self.Battery:
            battery_control = control.get("Battery", {})
            power_p_set = battery_control.get("P Setpoint", 0)
            t_battery_control = perf_counter()
            # battery_status = self.Battery.update(
            #     power_setpoint_W=power_p_set,
            #     timestamp=t,
            # )
            battery_soc, battery_kw, battery_set_kw, = self.Battery.update(
                power_setpoint_W=power_p_set,
                timestamp=t,
            )
            t_battery_update = perf_counter()
            # battery_kw = float(battery_status.get("Battery Electric Power (kW)", 0.0) or 0.0)
            # battery_set_kw = float(battery_status.get("Battery Set Power (kW)", 0.0) or 0.0)
            # battery_soc = float(battery_status.get("Battery SOC (-)", 0.0) or 0.0)

            # print(battery_soc, battery_kw, battery_set_kw,)
            self.battery_kw_arr[i] = battery_kw
            self.battery_set_kw_arr[i] = battery_set_kw
            self.battery_soc_arr[i] = battery_soc
            t_battery_update_info = perf_counter()
        t_battery = perf_counter()

        # ---------------- Power balance ----------------
        total_load_kw = demand_kw + ev_kw + thermal_kw
        total_generation_kw = pv_kw - battery_kw
        net_active_kw = round(total_load_kw - total_generation_kw, 3)

        self.total_kw_arr[i] = net_active_kw
        self.total_kvar_arr[i] = net_active_kw

        t_balance = perf_counter()

        out = {
            "Time": t,
            "Demand Electric Power (kW)": demand_kw,
            "PV Electric Power (kW)": pv_kw,
            "EV Electric Power (kW)": ev_kw,
            "EV Set Power (kW)": ev_set_kw,
            "EV SOC (-)": ev_soc,
            "EV Parked": ev_parked,
            "User Plug Out Time": user_plug_out_time,
            "Expected SOC": expected_soc,
            "Battery Electric Power (kW)": battery_kw,
            "Battery Set Power (kW)": battery_set_kw,
            "Battery SOC (-)": battery_soc,
            "HVAC Heating Electric Power (kW)": thermal_kw,
            "Temperature - Indoor (C)": indoor_temp,
            "Temperature - Outdoor (C)": float(self.outdoor_temp_arr[i]) if not np.isnan(
                self.outdoor_temp_arr[i]) else np.nan,
            "Total Electric Power (kW)": net_active_kw,
            "Total Reactive Power (kVAR)": net_active_kw,
        }

        t_return = perf_counter()

        print(
            f"base={(t_base - t0) * 1000:.3f} ms | "
            f"EV={(t_ev - t_base) * 1000:.3f} ms | "
            f"thermal={(t_thermal - t_ev) * 1000:.3f} ms | "
            f"battery={(t_battery - t_thermal) * 1000:.3f} ms | "
            f"battery control ={(t_battery_control - t_thermal) * 1000:.3f} ms | "
            f"battery update={(t_battery_update - t_battery_control) * 1000:.3f} ms | "
            f"battery info={(t_battery_update_info - t_battery_update) * 1000:.3f} ms | "

            f"balance={(t_balance - t_battery) * 1000:.3f} ms | "
            f"return_dict={(t_return - t_balance) * 1000:.3f} ms | "
            f"total={(t_return - t0) * 1000:.3f} ms"
        )

        return out
    def step(self, control_signal: dict | None) -> tuple[InverterModel, MeterModel, EVModel, HVACModel, dict]:
        """
        Main external simulator step method.

        Returns the same tuple structure as the original simulator:
            InverterModel, MeterModel, EVModel, HVACModel, house_status
        """
        try:

            house_status = self.update(control_signal)
            # house_status = self.update_profiled(control_signal)

        except StopIteration as exc:
            raise SimulationEnded("Simulation Ended -> add days to sim") from exc

        i = self.step_idx
        time_value = house_status["Time"]

        # ---------------- Inverter model ----------------
        self.InverterModel.time = time_value
        self.InverterModel.pv_power = float(house_status.get(INVERTER[0], 0.0))
        self.InverterModel.battery_power = float(house_status.get(INVERTER[1], 0.0))
        self.InverterModel.battery_soc = float(house_status.get(INVERTER[2], 0.0))

        # ---------------- Meter model ----------------
        self.MeterModel.time = time_value
        self.MeterModel.active_power = float(house_status.get(NET_POWER[0], 0.0))
        self.MeterModel.reactive_power = float(house_status.get(NET_POWER[1], 0.0))

        self.MeterModel.tariff = float(self.tariff_arr[i])
        self.MeterModel.feed_tariff = float(self.feed_tariff_arr[i])
        self.MeterModel.tariff_24hrs, self.MeterModel.feed_tariff_24hrs = self.get_next_24h_tariffs_fast(i)
        self.MeterModel.add_period(self.resolution.total_seconds() / 60.0)

        # ---------------- EV model ----------------
        self.EVModel.time = time_value
        self.EVModel.ev_status = bool(house_status.get(EV[0], 0.0))
        if self.EVModel.ev_status:
            self.EVModel.ev_soc = float(house_status.get(EV[1], 0.0))
            self.EVModel.user_ev_dc_time = house_status.get(EV[3], 0)
            self.EVModel.expected_soc = int(house_status.get(EV[4], 0) or 0)
        else:
            self.EVModel.ev_soc = 0.0
        self.EVModel.ev_power = float(house_status.get(EV[2], 0.0))

        # ---------------- HVAC model ----------------
        self.HVACModel.time = time_value
        self.HVACModel.ti = float(house_status.get(HVAC[0], 0.0))
        self.HVACModel.hvac_power = float(house_status.get(HVAC[1], 0.0))

        return self.InverterModel, self.MeterModel, self.EVModel, self.HVACModel, house_status

    # ---------------------------------------------------------------------
    # Export / logging
    # ---------------------------------------------------------------------
    def to_dataframe(self, update_internal: bool = True) -> pd.DataFrame:
        """
        Convert runtime arrays back to a pandas DataFrame.

        Parameters
        ----------
        update_internal:
            If True, self.simulation_df is replaced with the exported dataframe.
        """
        df = pd.DataFrame(index=self.time_index)
        df.index.name = "Time"

        df["Demand Electric Power (kW)"] = self.demand_kw_arr
        df["PV Electric Power (kW)"] = self.pv_kw_arr
        df["Total Electric Power (kW)"] = self.total_kw_arr
        df["Total Reactive Power (kVAR)"] = self.total_kvar_arr

        df["EV Electric Power (kW)"] = self.ev_kw_arr
        df["EV Set Power (kW)"] = self.ev_set_kw_arr
        df["EV SOC (-)"] = self.ev_soc_arr
        df["EV Parked"] = self.ev_parked_arr
        df["User Plug Out Time"] = self.ev_plug_out_time_arr
        df["Expected SOC"] = self.ev_expected_soc_arr

        df["Battery Electric Power (kW)"] = self.battery_kw_arr
        df["Battery Set Power (kW)"] = self.battery_set_kw_arr
        df["Battery SOC (-)"] = self.battery_soc_arr

        df["HVAC Heating Electric Power (kW)"] = self.hvac_kw_arr
        df["Temperature - Indoor (C)"] = self.indoor_temp_arr
        df["Temperature - Outdoor (C)"] = self.outdoor_temp_arr

        # Preserve other input columns from the original dataframe if they exist.
        for col in self.simulation_df.columns:
            if col not in df.columns:
                df[col] = self.simulation_df[col].to_numpy(copy=True)

        if update_internal:
            self.simulation_df = df

        return df

    def reset(self) -> None:
        """
        Reset runtime pointer to the beginning.

        Note: this does not reset internal state of ESS/EV/Thermal handlers. If you need a
        fully fresh episode for RL, recreate the simulator or add reset methods to those handlers.
        """
        self.step_idx = -1
        self.now_time = None


# Backward-compatible alias if your previous imports expect `dwelling`.
dwelling = DwellingFast
