import time
import pandas as pd
from datetime import datetime, timedelta
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)

from SRC.Controller.Database import PandasDatabase
from SRC.Controller.HEMSControlLib import HEMSController
import matplotlib.pyplot as plt
RESOLUTION = timedelta(minutes=15)  # 1 min resolution info
DURATION = timedelta(days=300)
START_TIME = datetime(2018, 1, 1)

# demand_config = {
#     'model': 'normal',
#     'file': './SRC/SIM/Defaults/Demand/15_min_normal_test.csv'
# }

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
House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
# Initialized House
House.initialized_df()


# # Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=timedelta(minutes=30),
                            ess_update_period= timedelta(minutes=30),
                            havc_update_period=timedelta(minutes=5))
#
# ########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = time.time()

while current_time <= end_time:
    inverter, meter, ev, hvac, status = House.step(control_signal)

    # get state for each controller
    # get action
    # get state (t+1)
    #
    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    if control_signal:
        # print(control_signal)
        pass
    # changing loop time
    current_time += RESOLUTION
    # if 24 hrs update tariff like in real case scenario
    # when and how will the tariff be updated
    # and if there is gap then how will the tariff be handled

end = time.time()
plt.figure(1)
plt.savefig('Cost per kwh.png')  # Save as PNG, PDF, SVG, etc.
plt.show()
plt.figure(2)
plt.hist(Controller.ev_controller.final_soc_list)
plt.savefig('final_soc.png')  # Save as PNG, PDF, SVG, etc.
plt.show()
plt.figure(3)
plt.hist(Controller.ev_controller.initial_soc_list)
plt.savefig('initial_soc.png')  # Save as PNG, PDF, SVG, etc.
plt.show()
print(f"Simulation took {end - start:.4f} seconds")
print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy}')
print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy}')
ev_soc = ev_config.get("capacity Wh")
print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy * ev_config.get("capacity Wh")}')
print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy * ev_config.get("capacity Wh")}')
print(f'Not fill charge count: {Controller.ev_controller.not_full_count}')

print(f'Final Cost: {Controller.hems_database.df["Instant Cost"].sum()}')
Controller.hems_database.df.to_csv('./Results/controller_test.csv')
House.simulation_df.to_csv('./Results/simulation_test.csv')
