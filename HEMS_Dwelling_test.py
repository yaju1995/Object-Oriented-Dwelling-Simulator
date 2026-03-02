import time
import pandas as pd
import random
from datetime import datetime, timedelta, time
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)

# from SRC.Controller.HEMSControlRL import HEMSController
from SRC.Controller.HEMSControlRule import HEMSController


RESOLUTION = timedelta(minutes=15)  # 1 min resolution info
DURATION = timedelta(days=30)
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
                 seed=1)

# to enable step to get inverter, meter, Hvac, ev information separately
# House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
# House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
# TOU tariff
House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_feed_tariff_example-TOU.csv')

# add TOU tariff day night and peak tariff
# Initialized House
House.initialized_df()


# # Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=timedelta(minutes=15),
                            ess_update_period= timedelta(minutes=15),
                            havc_update_period=timedelta(minutes=15),
                            mode='Test',
                            ev_config=ev_config,
                            ess_config=battery_config,
                            hvac_config=thermal_config)


#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = datetime.now()

### Train the moodels - 300 days
## Save the models - Properly name them
# Test the models - test them
# load the model before running
SEED = 0
TEST_EPS = 300
# Controller.load_models(episode=TEST_EPS)
random.seed(SEED)
day = 0
# Running a training loop
while current_time <= end_time:
    inverter, meter, ev, hvac, status = House.step(control_signal)

    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    if control_signal:
        # print(control_signal)
        pass
    # if current_time.time() == time(0, 0):
    #     Controller.hvac_controller.temp_ref = random.randrange(15, 26)  # ref is set

    # Updating time
    current_time += RESOLUTION

end = datetime.now()
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
# print(f'UnSatisfied SOC Wh: {Controller.ev_controller.unsatified_energy * ev_config.get("capacity Wh")}')
# print(f'Satisfied SOC Wh: {Controller.ev_controller.satisfied_energy * ev_config.get("capacity Wh")}')
# print(f'Not fill charge count: {Controller.ev_controller.not_full_count}')
# print(f'EV only charging cost : {Controller.ev_controller.total_ev_charging_cost}')
# print(f'EV only charging energy : {Controller.ev_controller.total_ev_charging_energy}')
# print(f'EV only total $/kwh : {Controller.ev_controller.total_ev_charging_cost/Controller.ev_controller.total_ev_charging_energy}')
#
# print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')

# # print('')
Controller.hems_database.df.to_csv('./Results/controller_test-TOU_ref.csv')
House.simulation_df.to_csv('./Results/simulation_test-TOU_ref.csv')

#
# Controller.hems_database.df.to_csv(f'./Results/controller_test-TOU_{TEST_EPS}_bound_2_change_t_ref_3.csv')
# House.simulation_df.to_csv(f'./Results/simulation_test-TOU_{TEST_EPS}_bound_2_change_t_ref_3.csv')


# REF RULE
# Controller.hems_database.df.to_csv(f'./Results/controller_test-dynamic_{TEST_EPS}_bound_2_change_t_ref_3.csv')
# House.simulation_df.to_csv(f'./Results/simulation_test-dynamic_{TEST_EPS}bound_2_change_t_ref_3.csv')