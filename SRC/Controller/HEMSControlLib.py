import pandas as pd
from datetime import timedelta
from torch import nn

from SRC.SIM.Tariff.tariffHandler import tariffHandler
from SRC.SIM.EquipmentClass import InverterModel, EVModel, HVACModel, MeterModel
from SRC.SIM.ControlSignalHandler import ControlSignal
from SRC.Controller.Database.PandasDatabase import DataStore
from .Constants import COLUMNS_KEYS

# from SRC.Controller.DDPGmodel.DDPG_Agent import DDPGAgent, DDPGConfig
from SRC.Controller.DDPGmodel.DDPG_Agent_n_step import DDPGAgent, DDPGConfig
from SRC.Controller.EV_controller.evControlLib import ev_controller
from SRC.support.lib_config import CustomLogger

logger = CustomLogger(command=True)

# EV controller design
EV_DDPG_config = DDPGConfig()
EV_DDPG_config.gamma = 0.9
EV_DDPG_config.actor_lr = 1e-3
EV_DDPG_config.critic_lr = 1e-3
EV_DDPG_config.hidden = (128, 128)
EV_DDPG_config.activation = (nn.ReLU, nn.Sigmoid)
# EV_RL_AGENT = DDPGAgent(obs_dim=8, act_dim=1,  cfg=EV_DDPG_config,n_step=1)
EV_RL_AGENT = DDPGAgent(obs_dim=8, act_dim=1,  cfg=EV_DDPG_config,n_step=1)

# ESS controller design
ESS_DDPG_Config = DDPGConfig()
ESS_DDPG_Config.hidden = (64, 64)
ESS_DDPG_Config.activation = (nn.ReLU, nn.Sigmoid)
ESS_RL_AGENT = DDPGAgent(obs_dim=5, act_dim=1,  cfg=ESS_DDPG_Config)

# Currently direct definition for early traning and testing

class HEMSController:
    def __init__(self, name: str,
                 data_resolution: timedelta,
                 meter_tariff: tariffHandler,
                 ev_tariff: tariffHandler = None,
                 ess_update_period:timedelta = timedelta(minutes=30),
                 ev_update_period:timedelta = timedelta(minutes=15),
                 havc_update_period:timedelta = timedelta(minutes=5)):
        """

        :param name: name for the controller
        :param data_resolution: timedelta (data storage resolution(1 min in most case))
        :param meter_tariff: global tariff for dwelling simulation
        :param ev_tariff: [optional], provide when ev tariff is separate from the meter_tariff
        """
        self.name = name
        self.resolution: timedelta = data_resolution
        self.ev_update_period = ev_update_period
        self.ess_update_period = ess_update_period
        self.hvac_update_period = havc_update_period
        self.hems_database = DataStore(resolution=data_resolution)
        self.hems_logs = pd.DataFrame(columns=COLUMNS_KEYS)
        self.hems_logs.index.name = "timestamp"
        self.control_signals = ControlSignal()

        self.ev_controller = ev_controller(EV_RL_AGENT, resolution=self.resolution,
                                           update_period=self.ev_update_period,
                                           global_database=self.hems_database)
        if self.ev_controller is not None:
            if ev_tariff is None:
                self.ev_controller.tariff_handler = meter_tariff
            else:
                self.ev_controller.tariff_handler = ev_tariff

        self.ess_controller = None
        self.hvac_controller = None
        self.load_forecasting_model = None
        self.generation_forecasting_model = None
        self.consumptionDataHandler = None  # Uncontrollable load, EV Load, HVAC LOad, Total LOad
        self.generationDataHandler = None  # PV generation
        self.storageDataHandler = None
        self.tariffHandler = None

        self.ESS_charge = False
        self.HVAC_ON = False

    def update(self, ev_info: EVModel, inverter_info: InverterModel, hvac_info: HVACModel, meter_info: MeterModel):

        now_time = meter_info.time
        # logger.commandline(f'Now time:{now_time}')
        # Consumption power
        consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
        hours = self.resolution.total_seconds() / 3600
        minute = self.resolution.total_seconds() / 60

        # Cost of consumed power
        instant_cost = round(meter_info.active_power * hours * meter_info.tariff, 4)

        # Storing information in HVAC
        row = {  # Base on Constants COLUMNS_KEYS
            'Consumption (kW)': consumption,
            'Consumption (kWh)': consumption * hours,
            'Generation (kW)': inverter_info.pv_power,
            'Generation (kWh)': inverter_info.pv_power * hours,
            'Total Electric Power (kW)': meter_info.active_power,
            'Total Electric Power (kWh)': meter_info.active_power * hours,

            'tariff': meter_info.tariff,
            'feed tariff': meter_info.feed_tariff,
            'Instant Cost': instant_cost,

            'Battery SOC (-)': inverter_info.battery_soc,
            'Battery Set Power (kW)': self.control_signals.Battery_P_Setpoint or 0,

            'EV Parked': ev_info.ev_status,
            'EV SOC (-)': ev_info.ev_soc,
            'EV Set Point (kW)': self.control_signals.EV_Max_Power or 0,
            'EV Consumption (kW)': ev_info.ev_power,
            'EV Consumption (kWh)': ev_info.ev_power * hours,

            'Temperature - Indoor (C)': hvac_info.ti,
            'HVAC Consumption (kW)': hvac_info.hvac_power,
            'HVAC Consumption (kWh)': hvac_info.hvac_power * hours,
        }
        next_24hr_tariff = meter_info.tariff_24hrs
        next_24hr_feed_tariff = meter_info.feed_tariff_24hrs

        # Database test
        self.hems_database.append(now_time, row)  # First update row then collect information

        # EV control
        self.control_signals.EV_Max_Power = self.ev_controller.update_status(ev_info=ev_info)
        # if ev_info.ev_status:
        #     self.control_signals.EV_Max_Power = 7
        # else:
        #     self.control_signals.EV_Max_Power = 0

        if self.control_signals.EV_Max_Power is not None:
            self.control_signals.EV_Max_Power = float(self.control_signals.EV_Max_Power) * 1000
        # HVAC Control
        if hvac_info.ti <= 20:
            self.HVAC_ON = True
        elif hvac_info.ti >= 25:
            self.HVAC_ON = False

        if self.HVAC_ON:
            self.control_signals.HVAC_Heating_Power = -8000
        else:
            self.control_signals.HVAC_Heating_Power = None
        # Battery control
        if inverter_info.battery_soc <= 0.0:
            self.ESS_charge = True
        elif inverter_info.battery_soc >= 0.8:
            self.ESS_charge = False

        if self.ESS_charge:
            self.control_signals.Battery_P_Setpoint = 2000
        else:
            self.control_signals.Battery_P_Setpoint = -consumption * 1000



        # if do_ev_update:
        #     df15 = self.hems_database.past_period_resampled(
        #         now_time, past_period=3,
        #         out_resolution=timedelta(minutes=15),
        #         headers=["Consumption (kWh)", "Generation (kWh)"],
        #         agg="sum",
        #     )
        #     if df15 is not None:
        #         print(f'now time : {now_time}')
        #         print(df15)

        # generate controller signals
        control_signal = self.control_signals.generate_control_signal()
        # logger.commandline(control_signal)
        return control_signal

    def get_past_period_df(self, now_time, past_period: int = 24):  # hope this will handle the resilution as well
        """
        Return exactly N samples ending at now_time, where:
          N = past_period(hours) / data_resolution

        Example:
          data_resolution = 1 minute, past_period=3 hours -> 180 rows
        """

        now_time = pd.Timestamp(now_time)
        res = self.resolution  # e.g. 1 min
        period = pd.Timedelta(hours=past_period)

        # how many rows we expect
        expected_len = int(period / res)  # 3h/1min = 180

        # Ensure now_time is aligned to resolution grid (optional but helpful)
        now_aligned = now_time.floor(res)

        # Pull a window (a bit larger than needed) then trim by length
        cutoff = now_aligned - period
        recent = self.hems_logs.loc[cutoff:now_aligned]

        if recent.empty:
            logger.commandline("Fallback: no data in requested window")
            return None

        # Take exactly the last expected_len rows
        if len(recent) < expected_len:
            logger.commandline(f"Fallback: insufficient rows (need {expected_len}, got {len(recent)})")
            return None

        control_data = recent.tail(expected_len)

        # ---- Optional strict checks ----
        # 1) Must end at now_aligned
        if control_data.index[-1] != now_aligned:
            logger.commandline(f"Fallback: last timestamp is {control_data.index[-1]} not {now_aligned}")
            return None

        # 2) Must be continuous at resolution (no missing minutes)
        expected_index = pd.date_range(
            end=now_aligned, periods=expected_len, freq=res
        )
        if not control_data.index.equals(expected_index):
            logger.commandline("Fallback: missing timestamps / irregular index in the last window")
            return None

        return control_data

    def get_resampled(
            self,
            df_sample: pd.DataFrame,
            resolution: timedelta,
            headers: list[str],
            agg: str = "mean",
    ) -> pd.DataFrame | None:
        """
        Resample the DataFrame to a given resolution for specific headers.
        Returns  extra value,

        resolution : datetime.timedelta (e.g., timedelta(minutes=15))
        headers    : list of column names to include
        agg        : aggregation function ('mean', 'sum', 'max', 'min')
        """

        if df_sample is None or df_sample.empty:
            return None

        # ---- Validate datetime index ----
        if not isinstance(df_sample.index, pd.DatetimeIndex):
            raise ValueError("df_sample must have a DatetimeIndex for resampling")

        # ---- Validate headers ----
        missing = set(headers) - set(df_sample.columns)
        if missing:
            raise KeyError(f"Missing columns in df_sample: {missing}")

        # ---- Convert timedelta to pandas frequency ----
        freq = f"{int(resolution.total_seconds())}s"

        # ---- Select requested headers only ----
        df = df_sample[headers]

        # ---- Apply aggregation ----
        if agg == "mean":
            return df.resample(freq).mean()
        elif agg == "sum":
            return df.resample(freq).sum()
        elif agg == "max":
            return df.resample(freq).max()
        elif agg == "min":
            return df.resample(freq).min()
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")
