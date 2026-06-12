import time
import random
from datetime import datetime, timedelta, time

import pandas as pd
import matplotlib.pyplot as plt

from SRC.SIM.Simulator_fast import dwelling, SimulationEnded
# from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list_train import (
    pv_config,
    ev_config,
    thermal_config,
    weather_file,
    demand_config,
    battery_config,
)

from SRC.Controller.HEMSControlRL import HEMSController
# from SRC.Controller.HEMSControlRule import HEMSController

from SRC.SIM.Tariff.TariffGenerator import RandomTariffGenerator

# ============================================================
# Simulation settings
# ============================================================

RES = 15
RESOLUTION = timedelta(minutes=RES)
DURATION = timedelta(days=5000)
START_TIME = datetime(2018, 1, 1, 0, 0)

SEED = 0
random.seed(SEED)

RESULTS_DIR = "./Results"
CONTROLLER_RESULTS_FILE = f"{RESULTS_DIR}/controller_train_ESS_tarriff.csv"
SIMULATION_RESULTS_FILE = f"{RESULTS_DIR}/simulation_train_ESS_tariff.csv"


# ============================================================
# Helper functions for the fast simulator
# ============================================================

def get_next_forecast_from_arrays(house: dwelling, default_value: float = 0.0) -> tuple[float, float]:
    """
    Get next-step demand and PV generation forecast from the fast simulator arrays.

    In Simulator_fast, self.simulation_df is mainly for setup/export.
    During runtime, use NumPy arrays and the current step index.
    """
    next_i = house.step_idx + 1

    if next_i >= house.n_steps:
        return default_value, default_value

    demand = float(house.demand_kw_arr[next_i])
    generation = float(house.pv_kw_arr[next_i])

    return demand, generation


def set_battery_soc_and_sync_array(house: dwelling, soc: float) -> None:
    """
    Reset battery SOC in the battery handler.

    The handler state is what matters for future battery dynamics. If the simulator also has
    a battery_soc_arr, update the current index for consistency in exported logs.
    """
    if house.Battery is None:
        return

    new_soc = house.Battery.set_soc(soc)

    if hasattr(house, "battery_soc_arr") and house.step_idx >= 0:
        # print(new_soc)
        house.battery_soc_arr[house.step_idx] = float(new_soc)


# ============================================================
# Create dwelling
# ============================================================

House = dwelling(
    name="Dwelling_1",
    start_time=START_TIME,
    resolution=RESOLUTION,
    duration=DURATION,
    demand_config=demand_config,
    weather_file=None,
    pv_config=pv_config,
    battery_config=battery_config,
    ev_config=ev_config,
    thermal_config=thermal_config,
    seed=SEED,
)

# ============================================================
# Tariff setup
# ============================================================

Tariff_gen = RandomTariffGenerator(
    low=0.1,
    high=0.4,
    resolution=RESOLUTION,
    seed=SEED,
)

House.tariff.tariff_model = Tariff_gen
House.tariff.feed_tariff_model = Tariff_gen

House.tariff.generate_tariff()
House.tariff.updated_tariff()

# ============================================================
# Initialise simulator
# ============================================================

House.initialized_df()

# ============================================================
# Controller setup
# ============================================================

Controller = HEMSController(
    name="Dwelling_1",
    data_resolution=RESOLUTION,
    meter_tariff=House.tariff,
    ev_update_period=RESOLUTION,
    ess_update_period=RESOLUTION,
    havc_update_period=RESOLUTION,
    ev_config=ev_config,
    ess_config=battery_config,
    hvac_config=thermal_config,
)

# Controller.load_models()
# Controller.load_models(episode=1000)


# ============================================================
# Training / simulation loop
# ============================================================
#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION

control_signal = {}
ev_status = False
day = 0

start = datetime.now()
day_start = datetime.now()
save_model = False

while current_time <= end_time - RESOLUTION:
    inverter, meter, ev, hvac, status = House.step(control_signal)

    step_end = datetime.now()

    current_time = status["Time"]

    # ------------------------------------------------------------
    # Forecast for next control period [add forecasting model]
    # ------------------------------------------------------------
    # need to add forecasting model [MPC controller + MILP]

    demand_forecast, generation_forecast = get_next_forecast_from_arrays(House)

    inverter.forecast_demand = demand_forecast
    inverter.forecast_generation = generation_forecast

    # ------------------------------------------------------------
    # Controller update
    # ------------------------------------------------------------
    control_signal = Controller.update(
        ev_info=ev,
        inverter_info=inverter,
        hvac_info=hvac,
        meter_info=meter,
    )
    current_time += RESOLUTION
    # ------------------------------------------------------------
    # Tariff update logic
    # ------------------------------------------------------------

    if current_time.time() == time(12, 0):
        # In a realistic setting, this represents getting the next-day tariff
        House.tariff.generate_tariff()
        # print(House.tariff.next_24hr_tariff)

    elif current_time.time() == time(0, 0) and current_time != START_TIME:
        # In a realistic setting, this applies the next-day tariff
        House.tariff.updated_tariff()

        day_end = datetime.now()

        # Reset battery SOC externally for training
        set_battery_soc_and_sync_array(House, random.uniform(0.05, 1.0))

        day += 1
        day_start = datetime.now()
        save_model = True

    # ------------------------------------------------------------
    # Progress display
    # ------------------------------------------------------------
    total_days = max(1, int(DURATION.total_seconds() // timedelta(days=1).total_seconds()))
    percent = min(day / total_days * 100, 100)
    bar = "█" * int(percent / 5) + "-" * (20 - int(percent / 5))

    if day > 0 and (day % 5 == 0 or day == total_days):
        avg_reward = Controller.ess_controller.avg_reward  #getattr(Controller.ess_controller, "avg_reward", None)
        print(
            f"\rSeed {SEED} |{bar}| {percent:.1f}% completed "
            f"::{day}/{total_days} days:: avg_reward={avg_reward}",
            end="",
        )

    # ------------------------------------------------------------
    # Optional model saving
    # ------------------------------------------------------------
    if day in (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000):
        if save_model:
            Controller.save_models(day)
            save_model = False

# ============================================================
# Export results
# ============================================================

end = datetime.now()
duration = (end - start).total_seconds()

print(f"\nSimulation took {duration:.4f} seconds")

# Simulator_fast does not continuously update House.simulation_df.
# Use to_dataframe() for final simulation results.
simulation_results = House.to_dataframe()
controller_results = Controller.hems_database.to_pandas()
# controller_results = Controller.hems_database.df

# print(f"Final House Cost: {controller_results['Instant Cost'].sum()}")

controller_results.to_csv(CONTROLLER_RESULTS_FILE, index=False)
simulation_results.to_csv(SIMULATION_RESULTS_FILE)

print(f"Saved controller results to: {CONTROLLER_RESULTS_FILE}")
print(f"Saved simulation results to: {SIMULATION_RESULTS_FILE}")
