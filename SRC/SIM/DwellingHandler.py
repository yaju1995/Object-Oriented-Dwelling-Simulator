from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from SRC.SIM.Simulator_Config.config_list import pv_config,ev_config,battery_config,thermal_config,demand_config,weather_file
from SRC.SIM.Simulator import dwelling

class HomeSimulator:
    def __init__(self, config, resolution, start_time, duration):
        self.dwelling = dwelling('test',
                                 start_time=start_time,
                                 resolution=resolution,
                                 duration=duration,
                                 demand_config=config.get('demand_confgi'))
