import pandas as pd
from datetime import datetime, timedelta
import torch.nn as nn

from SRC.SIM.Simulator import dwelling
from SRC.Controller.DDPGmodel.DDPG_Agent import DDPGAgent, DDPGConfig
from SRC.SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)
from SRC.SIM.ControlSignalHandler import ControlSignal
from SRC.Controller.HEMSControlRL import HEMSController
from SRC.Controller.EV_controller.evControlLib import ev_controller

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
House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')

Controller = HEMSController(name=House.name,data_resolution=RESOLUTION)

# EV controller design
EV_DDPG_config = DDPGConfig()
EV_DDPG_config.gamma = 0.9
EV_DDPG_config.actor_lr = 1e-3
EV_DDPG_config.critic_lr = 1e-3
EV_DDPG_config.activation = (nn.ReLU, nn.Sigmoid)

DDPG_EV = DDPGAgent(obs_dim=5, act_dim=1, max_action=1, cfg=EV_DDPG_config)
ev_cont = ev_controller(DDPG_EV, resolution=RESOLUTION)
ev_cont.tariff_handler.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
ev_cont.tariff_handler.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')

House.initialized_df()

# Initializing the simulation to
control_signal = {}
HEMS_Control_Signals = ControlSignal()

# inverter, meter, ev, hvac, status = House.step(control_signal)  # getting first values
# print(status)
# steps = len(list(House.simulation_df.index))  # we run for the define simulation step

# all controller set to false
heating = False  # control heating

########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION

while current_time <= end_time:  # need to change this to timedelta for
    #########################################################################
    # Interact with simulator to collect information
    # each step first collect information
    inverter, meter, ev, hvac, status = House.step(control_signal)
    Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    ##########################################################################
    ######### Over information extraction and evaluation ####################
    # demand value is something that the control has to estimated on it own[Real case compare with actual is need]
    demand_value = round(meter.active_power - inverter.battery_power + inverter.pv_power, 3)
    surplus = inverter.pv_power - demand_value

    #########################################################################
    current_tariff = meter.tariff  # handle by t    he meter in the house
    current_feed_tariff = meter.feed_tariff  # handle by the meter in the house
    now_time = inverter.time
    # past n hrs info
    past_n_hrs = 24
    cutoff = now_time - timedelta(hours=past_n_hrs)
    # print(now_time, cutoff)
    recent = House.simulation_df.loc[cutoff:now_time]


    # # Direct ESS control
    if surplus > 0 and inverter.battery_soc < 1:
        # charge battery
        HEMS_Control_Signals.Battery_P_Setpoint = surplus
    elif surplus < 0 and inverter.battery_soc > 0.2:
        # support the load
        HEMS_Control_Signals.Battery_P_Setpoint = surplus / 2
    else:
        # battery do nothing
        HEMS_Control_Signals.Battery_P_Setpoint = 0
    # ESS contoller signal Update

    if ev.ev_status:
        if ev.ev_soc < 1:
            HEMS_Control_Signals.EV_Max_Power = 7_000
        else:
            HEMS_Control_Signals.EV_Max_Power = 0
    # EV Controller signal Update

    if hvac.ti < 20:
        heating = True
    if hvac.ti > 25:
        heating = False

    if heating:
        HEMS_Control_Signals.HVAC_Heating_Power = -8000
    else:
        HEMS_Control_Signals.HVAC_Heating_Power = 0
    # HVAC controller update

    control_signal = HEMS_Control_Signals.generate_control_signal()

    # control = {'Battery': {'P Setpoint': 1000},
    #            'EV': {'Max Power': ev_power},
    #            'HVAC Heating':{'P Setpoint': hvac_power}}

    # print(next_5_period_tariff)
    ########################################################################
    # collect all the information to take action
    # ev_state ={
    #     'traiff': 'next t period',
    #     'feed tariff': 'next t period',
    #     'soc': ev.ev_soc
    #     ''
    # }
    # print(ev.ev_power)
    # Define HOME ENERGY MANAGEMENT SYSTEM [HEMS] that collect required information
    # generate forcasting value
    # Interact with all appliance

    # ev_power = ev_cont.update_status(ev)  # pass state and handle the control, user specific
    # House.controller.EV_Max_Power = ev_power
    # control_signal = House.generate_control_signal()

    # print(current_time, inverter.time) # checking the sync of time between simulation and simulator
    current_time += RESOLUTION
Controller.hems_logs.to_csv('./Results/controller_test_DDPG_03_cost_test.csv')
House.simulation_df.to_csv('./Results/simulation_test_DDPG_03_cost_test.csv')
