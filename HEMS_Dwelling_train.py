import time
import pandas as pd
from datetime import datetime, timedelta, time
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)

from SRC.Controller.Database import PandasDatabase
from SRC.Controller.HEMSControlLib import HEMSController
from SRC.SIM.Tariff.TariffGenerator import RandomTariffGenerator
import matplotlib.pyplot as plt
RESOLUTION = timedelta(minutes=15)  # 1 min resolution info
DURATION = timedelta(days=2)
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
# House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
# House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')

Tariff_gen = RandomTariffGenerator(low=0.1, high=0.5,resolution=timedelta(minutes=60))
House.tariff.tariff_model = Tariff_gen
House.tariff.feed_tariff = Tariff_gen
House.tariff.generate_tariff() # First Generate
print(House.tariff.tariff)
House.tariff.updated_tariff() # Then Update
print(House.tariff.tariff)
# exit()
# Initialized House
House.initialized_df()


# # Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=timedelta(minutes=30),
                            ess_update_period= timedelta(minutes=30),
                            havc_update_period=timedelta(minutes=5))
#
#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = time()

### Train the moodels - 300 days
## Save the models - Properly name them
# Test the models - test them
# load the model before running

# Controller.load_models()

# Running a training loop
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
    # day time is 12 pm get next day tariff and update ot traiff handler
    # print(current_time.time())
    if current_time.time() == time(12,00):
        print(f'{current_time.time()}Noon: Getting next day tariff')
        House.tariff.generate_tariff()
    elif current_time.time() == time(0,0):
        print(f'{current_time.time()}: mid night update tariff')
        House.tariff.updated_tariff()
    # if 24 hrs update tariff like in real case scenario
    # when and how will the tariff be updated
    # and if there is gap then how will the tariff be handled



# Save the models
# Controller.save_models()
#
# end = time.time()
# plt.figure(1)
# plt.savefig('Cost per kwh.png')  # Save as PNG, PDF, SVG, etc.
# plt.show()
# plt.figure(2)
# plt.hist(Controller.ev_controller.final_soc_list)
# plt.title('Final SOC Distribution')
# plt.savefig('final_soc.png')  # Save as PNG, PDF, SVG, etc.
# plt.show()
# plt.figure(3)
# plt.hist(Controller.ev_controller.initial_soc_list)
# plt.title('Initial SOC Distribution')
# plt.savefig('initial_soc.png')  # Save as PNG, PDF, SVG, etc.
# plt.show()
# print(f"Simulation took {end - start:.4f} seconds")
# print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy}')
# print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy}')
# ev_soc = ev_config.get("capacity Wh")
# print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy * ev_config.get("capacity Wh")}')
# print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy * ev_config.get("capacity Wh")}')
# print(f'Not fill charge count: {Controller.ev_controller.not_full_count}')
#
# print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')
# print(f'EV only charging cost : {Controller.ev_controller.total_ev_charging_cost}')
# print(f'EV only charging energy : {Controller.ev_controller.total_ev_charging_energy}')
# print(f'EV only total $/kwh : {Controller.ev_controller.total_ev_charging_cost/Controller.ev_controller.total_ev_charging_energy}')
# # print('')
# Controller.hems_database.df.to_csv('./Results/controller_train.csv')
# House.simulation_df.to_csv('./Results/simulation_train.csv')
