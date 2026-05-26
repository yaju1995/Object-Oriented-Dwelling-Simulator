"""
Example runner for Simulator_fast.py.

Place this file beside your project root so the SRC imports work, for example:
    project_root/
        run_fast_simulator.py
        Simulator_fast.py
        SRC/

Then run:
    python run_fast_simulator.py
"""

import datetime as dt

from Simulator_fast import dwelling, SimulationEnded


def main():
    sim = dwelling(
        name="test_house",
        start_time=dt.datetime(2018, 1, 1, 0, 0),
        duration=dt.timedelta(days=2),
        resolution=dt.timedelta(minutes=15),
        demand_config=None,
        weather_file=None,
        pv_config={
            "type": "train",
            "capacity W": 4000,
            "efficiency": 0.9,
            "area per W": 1.0,
            "tilt": 30,
            "azimuth": 180,
            "max_irradiance": 1000,
        },
        battery_config={
            "capacity Wh": 10_000,
            "initial soc": 50,
            "charging power W": 5_000,
            "discharging power W": 5_000,
            "charging eff": 0.95,
            "discharging eff": 0.95,
        },
        ev_config=None,
        thermal_config=None,
        seed=42,
    )

    sim.initialized_df()

    while True:
        try:
            # Example control signal.
            # Positive/negative convention depends on your ESSHandler implementation.
            control = {
                "Battery": {"P Setpoint": 0},
                "EV": {"Max Power": 0},
                "HVAC Heating": {"P Setpoint": 0},
            }

            inverter, meter, ev, hvac, status = sim.step(control)

            # Print only first few steps to avoid flooding terminal.
            if sim.step_idx < 5:
                print(
                    status["Time"],
                    "Net kW=", status["Total Electric Power (kW)"],
                    "PV kW=", status["PV Electric Power (kW)"],
                    "Battery SOC=", status["Battery SOC (-)"],
                )

        except SimulationEnded:
            break

    results = sim.to_dataframe()
    results.to_csv("fast_simulation_results.csv")
    print("Saved: fast_simulation_results.csv")
    print(results.head())


if __name__ == "__main__":
    main()
