import pandas as pd
from datetime import timedelta
from torch import nn

from SRC.SIM.Tariff.tariffHandler import tariffHandler
from SRC.SIM.EquipmentClass import InverterModel, EVModel, HVACModel, MeterModel
from SRC.SIM.ControlSignalHandler import ControlSignal
from SRC.Controller.Database.PandasDatabase import DataStore
from .Constants import COLUMNS_KEYS
# from SRC.SIM.Simulator_Config.config_list import (ev_config, battery_config)

from SRC.Controller.DDPGmodel.DDPG_Agent_multistep import DDPGAgent, DDPGConfig
from SRC.Controller.DDPGmodel.bounded_DDPG_Agent_multistep import Bound_DDPGAgent
from SRC.Controller.DDPGmodel.DDGP_Bound_Agent_old import DPGAgent


from SRC.support.lib_config import CustomLogger
import os

logger = CustomLogger(command=True)

# EV Import and Definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rule based ~~~~~~~~~
from SRC.Controller.EV_controller.evRuleControlLib import ev_controller

# RL based ~~~~~~~
# from SRC.Controller.EV_controller.evControlLib import ev_controller

EV_DDPG_config = DDPGConfig()
EV_DDPG_config.gamma = 0.9
EV_DDPG_config.actor_lr = 1e-3
EV_DDPG_config.critic_lr = 1e-3
EV_DDPG_config.hidden = (128, 128)
EV_DDPG_config.activation = (nn.ReLU, nn.Sigmoid)
EV_DDPG_config.a_max = 1.0
EV_DDPG_config.a_min = 0.0
EV_LOOK_AHEAD = 4  # If you update this try to match the input state with this
EV_INPUT_DIM = 4 + EV_LOOK_AHEAD  # try to match the observed state with delay steps
EV_OUT_DIM = 1
EV_RL_AGENT = DDPGAgent(name='EVagent', obs_dim=EV_INPUT_DIM, act_dim=EV_OUT_DIM, cfg=EV_DDPG_config,
                        n_step=EV_LOOK_AHEAD,
                        return_mode='nstep')
EV_MODEL_DIR = f'Models/EV/test_res_15_{EV_LOOK_AHEAD}/'
EV_MODEL_NAME = f'states_{EV_INPUT_DIM}_config1_delay_{EV_LOOK_AHEAD}_nstep_cost_4.pth'

# ESS Import and definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rule based controller ~~~~~~~~~~~~~~~~
# from SRC.Controller.ESS_controller.ESSRuleControlLib import essController
# RL agent based controller ~~~~~~~~~~~~~~~~
from SRC.Controller.ESS_controller.essRLControlLib import essController
ESS_DDPG_config = DDPGConfig()
ESS_DDPG_config.gamma = 0.9
ESS_DDPG_config.actor_lr = 1e-3
ESS_DDPG_config.critic_lr = 1e-3
ESS_DDPG_config.hidden = (256,)
ESS_DDPG_config.activation = (nn.ReLU, nn.Tanh)
ESS_DDPG_config.batch_size = 1000
ESS_DDPG_config.a_max = 1.0
ESS_DDPG_config.a_min = -1.0
# ESS_DDPG_config.tau = 1
# ESS_DDPG_config.seed = 42
ESS_LOOK_AHEAD = 1  # If you update this try to match the input state with this
ESS_INPUT_DIM = 1 + ESS_LOOK_AHEAD*2  # try to match the observed state with delay steps
ESS_OUT_DIM = 1
ESS_RL_AGENT = Bound_DDPGAgent(name='ESSagent', obs_dim=ESS_INPUT_DIM, act_dim=ESS_OUT_DIM, cfg=ESS_DDPG_config,
                               n_step=ESS_LOOK_AHEAD,
                               return_mode='nstep')
# agent = DPGAgent(ESS_INPUT_DIM, ESS_OUT_DIM, 1, gamma=0.9, seed=0)
ESS_MODEL_DIR = f'Models/ESS/nstep_res_15_{ESS_LOOK_AHEAD}_diff_tariff/'
ESS_MODEL_NAME = f'states_{ESS_INPUT_DIM}_delay_{ESS_LOOK_AHEAD}.pth'
# mode name from other
# ESS_MODEL_NAME = f'seed_1_1000.pth'


# Currently direct definition for early training and testing

class HEMSController:
    def __init__(self, name: str,
                 data_resolution: timedelta,
                 meter_tariff: tariffHandler,
                 ev_tariff: tariffHandler = None, # pass ESS, EV and hvac control config from the simulator
                 ess_update_period: timedelta = timedelta(minutes=15),
                 ess_config:dict= None,
                 ev_update_period: timedelta = timedelta(minutes=15),
                 ev_config:dict= None,
                 havc_update_period: timedelta = timedelta(minutes=5),
                 hvac_config:dict = None,
                 mode='Train',
                 controller='RL'):
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
        # self.mode = mode
        # Databased definition
        self.hems_database = DataStore(resolution=data_resolution)
        self.hems_logs = pd.DataFrame(columns=COLUMNS_KEYS)
        self.hems_logs.index.name = "timestamp"
        self.control_signals = ControlSignal()
        self.controller = controller
        self.ev_controller = None
        self.ess_controller = None
        self.hvac_controller = None
        self.ev_config = ev_config
        self.ess_config = ess_config
        self.hvac_config = hvac_config
        # EV RL controller
        # self.ev_controller = ev_controller(rl_agent=EV_RL_AGENT, resolution=self.resolution,
        #                                    update_period=self.ev_update_period,
        #                                    global_database=self.hems_database, mode=mode,
        #                                    max_charging_power=ev_config.get('charging power W', 7_000) / 1000,
        #                                    look_ahead=EV_LOOK_AHEAD)
        # Ev Rule controller
        # self.ev_controller = ev_controller(resolution=self.resolution,
        #                                    update_period=self.ev_update_period,
        #                                    global_database=self.hems_database, mode=mode,
        #                                    max_charging_power=ev_config.get('charging power W', 7_000) / 1000,
        #                                    look_ahead=EV_LOOK_AHEAD)
        #
        # if self.ev_controller is not None:
        #     if ev_tariff is None:
        #         self.ev_controller.tariff_handler = meter_tariff
        #     else:
        #         self.ev_controller.tariff_handler = ev_tariff
        # ESS RL controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ess_controller = essController(rl_agent=ESS_RL_AGENT,
                                            mode=mode,
                                            resolution=self.resolution,
                                            update_period=self.ess_update_period,
                                            global_database=self.hems_database,
                                            max_charging_kw=self.ess_config.get('charging power W', 1000) / 1000,
                                            max_discharging_kw=self.ess_config.get("discharging power W", 1000) / 1000,
                                            look_ahead=ESS_LOOK_AHEAD,
                                            energy_normalizer=self.ess_config.get('capacity Wh') / 1000
                                            )
        # RULE based Controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.ess_controller = essController(resolution=self.resolution,
        #                                     update_period=self.ess_update_period,
        #                                     global_database=self.hems_database,
        #                                     max_charging_kw=ess_config.get('charging power W', 1000) / 1000,
        #                                     max_discharging_kw=ess_config.get("discharging power W", 1000) / 1000,
        #                                     look_ahead=ESS_LOOK_AHEAD,
        #                                     )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ess_controller.tariff_handler = meter_tariff

        self.load_forecasting_model = None
        self.generation_forecasting_model = None

        # Load models

        self.tariff_handler = None

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
            'Battery Set Power (W)': self.control_signals.Battery_P_Setpoint or 0,
            'Battery Power (kW)': inverter_info.battery_power,
            'Battery Energy (kWh)': inverter_info.battery_power * hours,

            'EV Parked': ev_info.ev_status,
            'EV SOC (-)': ev_info.ev_soc,
            'EV Set Point (kW)': self.control_signals.EV_Max_Power or 0,
            'EV Consumption (kW)': ev_info.ev_power,
            'EV Consumption (kWh)': ev_info.ev_power * hours,

            'Temperature - Indoor (C)': hvac_info.ti,
            'HVAC Consumption (kW)': hvac_info.hvac_power,
            'HVAC Consumption (kWh)': hvac_info.hvac_power * hours,
        }

        # Database test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.hems_database.append(now_time, row)  # First update row then collect information

        # EV control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.control_signals.EV_Max_Power = self.ev_controller.update_status(ev_info=ev_info)
        #
        # if self.control_signals.EV_Max_Power is not None:
        #     self.control_signals.EV_Max_Power = float(self.control_signals.EV_Max_Power) * 1000

        # HVAC control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if hvac_info.ti <= 20:
        #     self.HVAC_ON = True
        # elif hvac_info.ti >= 25:
        #     self.HVAC_ON = False
        #
        # if self.HVAC_ON:
        #     self.control_signals.HVAC_Heating_Power = -8000
        # else:
        #     self.control_signals.HVAC_Heating_Power = None

        # Battery control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.control_signals.Battery_P_Setpoint = self.ess_controller.update_status(meter_info=meter_info,
                                                                                    inverter_info=inverter_info)
        if self.control_signals.Battery_P_Setpoint is not None:
            self.control_signals.Battery_P_Setpoint = float(self.control_signals.Battery_P_Setpoint) * 1000

        # generate controller signals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        control_signal = self.control_signals.generate_control_signal()
        # logger.commandline(control_signal)
        # return the generated signal
        return control_signal

    def get_past_period_df(self, now_time, past_period: int = 24):  # hope this will handle the resolution as well
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

    def save_models(self):
        # if self.ev_controller:
        #     os.makedirs(EV_MODEL_DIR, exist_ok=True)
        #     EV_PATH = os.path.join(EV_MODEL_DIR, EV_MODEL_NAME)
        #     logger.commandline(EV_RL_AGENT.save(EV_PATH))
        if self.ess_controller:
            os.makedirs(ESS_MODEL_DIR, exist_ok=True)
            ESS_PATH = os.path.join(ESS_MODEL_DIR, ESS_MODEL_NAME)
            logger.commandline(self.ess_controller.rl_agent.save(ESS_PATH))
            # logger.commandline(agent.save(ESS_PATH))
        # save command for other models

    def load_models(self):
        # if self.ev_controller:
        #     EV_PATH = os.path.join(EV_MODEL_DIR, EV_MODEL_NAME)
        #     logger.commandline(EV_RL_AGENT.load(EV_PATH))
        if self.ess_controller:
            ESS_PATH = os.path.join(ESS_MODEL_DIR, ESS_MODEL_NAME)
            logger.commandline(self.ess_controller.rl_agent.load(ESS_PATH))
            # logger.commandline(agent.load(ESS_PATH))
