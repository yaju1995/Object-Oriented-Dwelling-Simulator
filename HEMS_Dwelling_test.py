import pandas as pd
from datetime import datetime, timedelta
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)
from SRC.Controller.HEMSControlLib import HEMSController

RESOLUTION = timedelta(minutes=1)  # 1 min resolution info
DURATION = timedelta(days=2)
START_TIME = datetime(2018, 1, 1)

House = dwelling(name='Dwelling_1',
                 start_time=START_TIME,
                 resolution=RESOLUTION,
                 duration=DURATION,
                 demand_config=demand_config,
                 weather_file=weather_file,
                 pv_config=pv_config,
                 battery_config=battery_config,
                 ev_config=ev_config,
                 thermal_config=thermal_config,
                 seed=0)

# to enable step to get inverter, meter, Hvac, ev information separately
House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
# Initialized House
House.initialized_df()

# Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff)

########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

EV_CONTROL_TIME = timedelta(minutes=15)
ESS_CONTROL_TIME = timedelta(minutes=15)
HVAC_CONTROL_TIME = timedelta(minutes=5)

while current_time <= end_time:
    inverter, meter, ev, hvac, status = House.step(control_signal)
    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    if control_signal:
        # print(control_signal)
        pass
    # GEt ESS control
    # get state
    # get control logic from agent
    # Pass control
    # GEt EV control
    # GEt HVAC Control

    # changing loop time
    current_time += RESOLUTION

Controller.hems_database.df.to_csv('./Results/controller_test.csv')
House.simulation_df.to_csv('./Results/simulation_test.csv')
