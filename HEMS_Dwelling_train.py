import time
import pandas as pd
import random
from datetime import datetime, timedelta, time
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list_train import (pv_config,
                                                        ev_config,
                                                        thermal_config,
                                                        weather_file,
                                                        demand_config,
                                                        battery_config)

from SRC.Controller.Database import PandasDatabase
from SRC.Controller.HEMSControlRL import HEMSController
from SRC.SIM.Tariff.TariffGenerator import RandomTariffGenerator
import matplotlib.pyplot as plt
RES = 15
RESOLUTION = timedelta(minutes=RES)  # 1 min resolution info
DURATION = timedelta(days=500)
START_TIME = datetime(2018, 1, 1)

# demand_config = {
#     'model': 'normal',
#     'file': './SRC/SIM/Defaults/Demand/15_min_normal_test.csv'
# }
SEED = 0
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
                 seed=SEED)

# to enable step to get inverter, meter, Hvac, ev information separately
# House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
# House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')

Tariff_gen = RandomTariffGenerator(low=0.1, high=0.4, resolution=timedelta(minutes=RES), seed=SEED)
House.tariff.tariff_model = Tariff_gen
House.tariff.feed_tariff_model = Tariff_gen
House.tariff.generate_tariff()  # First Generate
House.tariff.updated_tariff()  # Then Update
House.initialized_df()

# ev_controller_config = {
#     'update period': timedelta(minutes=15),
#     'ev_config': ev_config
# }
#
# ess_controller_config ={
#
# }

# # Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=timedelta(minutes=RES),
                            ess_update_period=timedelta(minutes=RES),
                            havc_update_period=timedelta(minutes=RES),
                            ev_config=ev_config,
                            ess_config=battery_config,
                            hvac_config=thermal_config)
#
#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = datetime.now()

### Train the moodels - 300 days
## Save the models - Properly name them
# Test the models - test them
# load the model before running

# Controller.load_models()
random.seed(SEED)
day = 0
# Running a training loop
while current_time <= end_time:

    inverter, meter, ev, hvac, status = House.step(control_signal)

    # get state for each controller
    # get action
    # get state (t+1)
    #
    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    # Update control parameter like, temperature, EV cost saving and so on
    # Controller.hvac_controller.temp_ref = 22.5
    # Controller.hvac_controller.temp_deviation = 2


    if control_signal:
        # print(control_signal)
        pass
    # changing loop time
    current_time += RESOLUTION
    # day time is 12 pm get next day tariff and update ot tariff handler
    # print(current_time.time())
    if current_time.time() == time(12, 00):
        print(f'{current_time.time()}: Noon: Getting next day tariff')
        House.tariff.generate_tariff()
    elif current_time.time() == time(0, 0):
        print(f'{current_time.time()}: Mid night update tariff')
        House.tariff.updated_tariff()
        # update the SOC external for training
        next_soc = House.Battery.set_soc(random.uniform(0.05, 1)) # reset that will occur
        day +=1
        print(f'{current_time}: day')
        # updating the temp ref point
        Controller.hvac_controller.temp_ref = random.randrange(15, 26) # ref is set
        # may be change the bound but let see first.



    # if 24 hrs update tariff like in real case scenario
    # when and how will the tariff be updated
    # and if there is gap then how will the tariff be handled
    # if day in (30,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950, 1000, 1250, 1500, 1750, 2000):
    if day in (300,350,400,450,500,550,600):
        Controller.save_models(day)
# Save the models
# Controller.save_models(day)

end = datetime.now()
duration = (end - start).total_seconds()
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
plt.show()
print(f"Simulation took {duration:.4f} seconds")
# print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy}')
# print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy}')
# ev_soc = ev_config.get("capacity Wh")
# print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy * ev_config.get("capacity Wh")}')
# print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy * ev_config.get("capacity Wh")}')
# print(f'Not fill charge count: {Controller.ev_controller.not_full_count}')
# print(f'EV only charging cost : {Controller.ev_controller.total_ev_charging_cost}')
# print(f'EV only charging energy : {Controller.ev_controller.total_ev_charging_energy}')
# print(
#     f'EV only total $/kwh : {Controller.ev_controller.total_ev_charging_cost / Controller.ev_controller.total_ev_charging_energy}')
# print('')
print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')

Controller.hems_database.df.to_csv('./Results/controller_train_change_ref.csv')
House.simulation_df.to_csv('./Results/simulation_train_change_ref.csv')
