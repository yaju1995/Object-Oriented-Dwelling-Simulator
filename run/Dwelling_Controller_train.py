from datetime import datetime, timedelta
from time import perf_counter
from SRC.SIM.Simulator_fast import dwelling

from config import ev_config, battery_config, thermal_config, weather_file, pv_config, demand_config
from SRC.SIM.ControlSignalHandler import ControlSignal

# ============================================================
# Simulation settings
# ============================================================

RES = 15
RESOLUTION = timedelta(minutes=RES)
DURATION = timedelta(days=500)
START_TIME = datetime(2018, 1, 1, 0, 0)

SEED = 0
t0 = perf_counter()
House = dwelling(
    name="Dwelling_1",
    start_time=START_TIME,
    resolution=RESOLUTION,
    duration=DURATION,
    demand_config=demand_config,
    weather_file=weather_file,
    pv_config=pv_config,
    battery_config=battery_config,
    ev_config=ev_config,
    thermal_config=thermal_config,
    seed=SEED,
)

House.initialized_df()

t_iniit = perf_counter()
# tariff definition: If not defined all values '0'
House.tariff.upload_tariff('../SRC/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
House.tariff.upload_feed_tariff('../SRC/SIM/Defaults/Tariff/hourly_feed_tariff_example-TOU_0_2.csv')

t_tariff = perf_counter()
######################################################################################################################

from run.Controller.HEMS_Controller.RULE_HEMSController import RuleController

Controller = RuleController(name='Rule', resolution=RESOLUTION, train=False,tariff_info= House.tariff)

######################################################################################################################
current_time = START_TIME
end_time = START_TIME + DURATION

controller = ControlSignal()
control_signal = controller.generate_control_signal()
ev_status = False
day = 0

start = datetime.now()
day_start = datetime.now()
save_model = False
t_controller = perf_counter()
while current_time <= end_time - RESOLUTION:
    inverter, meter, ev, hvac, status = House.step(control_signal)
    # print(f'{current_time}: {control_signal} : end time :{end_time}| tariff : {meter.tariff}')
    step_end = datetime.now()

    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, meter_info=meter, hvac_info=hvac)
    current_time += RESOLUTION


t_sim = perf_counter()

print(f'init = {(t_iniit-t0)*1000:.3f} ms|'
      f'tariff {(t_tariff-t_iniit)*1000:.3f} ms|'
      f'control {(t_controller-t_tariff)*1000:.3f} ms|'
      f'sim {(t_sim-t_controller)*1000:.3f} ms|')
RESULTS_DIR = "./Results"
SIMULATION_RESULTS_FILE = f"{RESULTS_DIR}/simulation_train_ESS_tariff.csv"
simulation_results = House.to_dataframe()
simulation_results.to_csv(SIMULATION_RESULTS_FILE)
print(f"Saved simulation results to: {SIMULATION_RESULTS_FILE}")
