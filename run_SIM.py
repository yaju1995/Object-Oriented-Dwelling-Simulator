import logging
import numpy as np

from SRC.SIM.Simulator import dwelling
from SRC.SIM.ESS.ess_handler import ESSHandler
from SRC.SIM.EV.ev_handler import EVHandler
from SRC.SIM.Thermal.thermal_handler import ThermalHandler

from datetime import datetime, timedelta

RESOLUTION = timedelta(minutes=15)
START_TIME = datetime(2018,1,1)
DURATION = timedelta(days=5)
##################################################################################################
# Defining battery storage

battery_config = {
    "capacity Wh": 5_000,
    "initial soc": 50,
    "charging power W": 2500,
    "discharging power W": 2500,
    "charging eff": 1,
    "discharging eff": 1,
}

ev_config = {
    "capacity Wh": 60_000,  # 60 kWh
    "charging power W": 7_400,  # 7.4 kW charger
    "discharging power W": 7_400,  # V2G capable
    "v2g_enabled": False,
    "profile_file":"./SRC/SIM/Defaults/EV/pdf_Veh1_Level0.csv"
}
thermal_config = {
    'initial temperature C': 21,
    'tau': 30 * 3600,
    'W': 200,
    'n': 0.95
}

demand_config = {
    'model': 'normal',
    'file': './SRC/SIM/Defaults/Demand/15_min_normal_test.csv'
}

weather_file = './SRC/SIM/Defaults/Weather/IRL_Dublin.039690_IWEC.epw'

pv_config = {
    "capacity kW": 5,
    "efficiency": 0.95,
    "area per kW":1,
    "tilt": 20,
    "azimuth": 0,
}

House = dwelling(name='H1',start_time=START_TIME,
                 duration=DURATION,
                 resolution=RESOLUTION,
                 demand_config=demand_config,
                 weather_file=weather_file,
                 pv_config=pv_config,
                 battery_config = battery_config,
                 ev_config=ev_config,
                 thermal_config=thermal_config)



##################################################################################################
# Over writing the existing models
# House = dwelling(name='H1',start_time=datetime(2018, 1, 1),
#                  duration=timedelta(days=10),
#                  resolution=RESOLUTION,weather_file= weather_file,
#                  demand_config=demand_config,
#                  pv_config=pv_config, battery_config = None, ev_config=None, thermal_config=None)

# pre defining overwriting configuration
# House.Battery = ESSHandler(name='ESS',
#                            resolution=timedelta(minutes=15),
#                            total_capacity_Wh=6_000,
#                            charging_power_W=3_000,
#                            discharging_power_W=3_000,
#                            initial_soc_pct=50,
#                            )
#
#
# House.Thermal = ThermalHandler(resolution=RESOLUTION,
#                               initial_internal_temperature=thermal_config.get(
#                               'initial temperature C'),
#                               tau=thermal_config.get('tau'),
#                               W=thermal_config.get('W'),
#                               n=thermal_config.get('n'))


##################################################################################################

House.initialized_df()
print(House.simulation_df.head())
print(House.simulation_df.keys())

NET_POWER = ['Total Electric Power (kW)', 'Total Reactive Power (kVAR)']
INVERTER = ['PV Electric Power (kW)', 'Battery Electric Power (kW)', 'Battery SOC (-)']
EV = ['EV Parked', 'EV SOC (-)', 'EV Electric Power (kW)']
HVAC = ["Temperature - Indoor (C)", "HVAC Heating Electric Power (kW)", "Temperature - Outdoor (C)"]

# print(House.simulation_df.head())
# print(House.simulation_df.tail())
# print(House.simulation_df.keys())

control = {}
status = House.update(control)
heating = False
print(status)
steps = len(list(House.simulation_df.index))

for t in range(steps - 1):
    # if ev detected enable ev agent
    ev_power = 0
    hvac_power = 0
    if House.Thermal:
        if 21>status.get(HVAC[0]):
            heating = True
        elif status.get(HVAC[0])>25:
            heating = False
    if heating:
        hvac_power = -8000

    if status.get(EV[0]):
        ev_power = 5000
    # Thermal control enable
    # get action based on the first state
    control = {'Battery': {'P Setpoint': 1000},
               'EV': {'Max Power': ev_power},
               'HVAC Heating':{'P Setpoint': hvac_power}}
    status = House.update(control)

    # print(status.get(INVERTER[1], 0))
    # print(status.get(INVERTER[2], 0))
#
House.simulation_df.to_csv('./Results/simulation_test.csv')
# House.upload_data('./SRC/SIM/Example/update_sample.csv')

# House.upload_data('Active_power_kw', file)
# House.upload_data('Ambient_temperature_C', file)
# print(House.simulation_df.head())
# print(House.simulation_df.tail())
