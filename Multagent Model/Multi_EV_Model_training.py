import random
from datetime import datetime, timedelta, time
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list_train import ev_config, BASE_DIR
from SRC.SIM.Tariff.tariffHandler import tariffHandler
from SRC.SIM.Tariff.TariffGenerator import RandomTariffGenerator
from SRC.SIM.ControlSignalHandler import ControlSignal
from MultiHEMSControlRL import HEMSController
from EV_info_extractor import extract_controller_metrics,extract_ev_connection_metrics
import pandas as pd

# print(BASE_DIR)
# print(ev_config["profile_file"])
RES = 15
RESOLUTION = timedelta(minutes=RES)  # 1 min resolution info
DURATION = timedelta(days=50)
START_TIME = datetime(2018, 1, 1)


SEED = 0
Tariff_gen = RandomTariffGenerator(
    low=0.1,
    high=0.4,
    resolution=timedelta(minutes=RES),
    seed=SEED
)

global_tariff = tariffHandler()
global_tariff.tariff_model = Tariff_gen
global_tariff.feed_tariff_model = Tariff_gen
global_tariff.generate_tariff()
global_tariff.updated_tariff()

# -----------------------------------------
# 2. Create multiple dwellings + controllers
# -----------------------------------------
NUM_HOUSES = 2
houses = []
controllers = []

for i in range(NUM_HOUSES):
    house = dwelling(
        name=f"Dwelling_{i+1}",
        start_time=START_TIME,
        resolution=RESOLUTION,
        duration=DURATION,
        demand_config=None,
        weather_file=None,
        pv_config=None,
        battery_config=None,
        ev_config=ev_config,
        thermal_config=None,
        seed=i  # different seed per house
    )

    # Assign shared tariff
    house.tariff = global_tariff
    house.initialized_df()

    houses.append(house)

    # Create controller for this house
    controller = HEMSController(
        name=f"Controller_{i+1}",
        data_resolution=RESOLUTION,
        meter_tariff=global_tariff,   # shared tariff
        ev_update_period=timedelta(minutes=RES),
        ess_update_period=timedelta(minutes=RES),
        havc_update_period=timedelta(minutes=RES),
        ev_config=ev_config,
        ess_config=None,
        hvac_config=None
    )

    controllers.append(controller)


#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = datetime.now()

### Train the moodels - 300 days
## Save the models - Properly name them
# Test the models - test them
# load the model before running
ev_status = False

# Controller.load_models()
random.seed(SEED)
day = 0
# Running a training loop
while current_time <= end_time:
    for i in range(NUM_HOUSES):
        inverter, meter, ev, hvac, status = houses[i].step(control_signal)
        control_signal = controllers[i].update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)


    if control_signal:
        # print(control_signal)
        pass
    # changing loop time
    current_time += RESOLUTION
    # day time is 12 pm get next day tariff and update ot tariff handler
    # print(current_time.time())
    if current_time.time() == time(12, 00):
        print(f'{current_time.time()}: Noon: Getting next day tariff')
        global_tariff.generate_tariff()
    elif current_time.time() == time(0, 0):
        print(f'{current_time.time()}: Mid night update tariff')
        global_tariff.updated_tariff()
        day += 1
        print(f'{current_time}: day')

    # changing EV disconnect time: is received from the simulator then randomly change with probability
    if ev_status == False:  # to initialize the user requirement condition with ev connection
        if ev.ev_status:
            ev_status = True
            # Use probability to update EV expected soc or EV disconnect time
    elif ev_status == True:
        if ev.ev_status == False:
            ev_status = False
        # else:
        #     House.EV.user_set_plugout = #change

end = datetime.now()
duration = (end - start).total_seconds()
summary_rows = []

for i, (house, controller) in enumerate(zip(houses, controllers), start=1):

    # Save raw data
    controller.hems_database.df.to_csv(f'./Results/controller_{i}.csv')
    house.simulation_df.to_csv(f'./Results/simulation_{i}.csv')

    # Extract metrics
    ctrl_metrics = extract_controller_metrics(controller, ev_capacity_Wh=ev_config["capacity Wh"])
    ev_conn_metrics = extract_ev_connection_metrics(house.simulation_df)

    # Combine into one row
    row = {
        "house_id": i,
        **ctrl_metrics,
        **ev_conn_metrics
    }

    summary_rows.append(row)

# Convert to summary dataframe
summary_df = pd.DataFrame(summary_rows)

# Save summary
summary_df.to_csv("./Results/summary_all_houses.csv", index=False)

