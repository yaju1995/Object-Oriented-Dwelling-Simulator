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
    "discharging power W": 0,  # V2G capable
    "charging eff": 1,
    "discharging eff": 1,
    "v2g_enabled": False,
    "profile_file":"./SRC/SIM/Defaults/EV/pdf_Veh1_Level0.csv"
}

thermal_config = {
    'initial temperature C': 21,
    'tau': 30 * 3600, # need to get reference for these values
    'W': 200, # need to get reference for these values
    'n': 0.95 # need to get reference for these values
}

# model type with list
# file path to the data for the model
demand_config = {
    'model': 'normal',
    'file': './SRC/SIM/Defaults/Demand/15_min_normal_test.csv'
}

# Weather file is the just Path ot name of the CSV with .epw
weather_file = './SRC/SIM/Defaults/Weather/IRL_Dublin.039690_IWEC.epw'
# weather_file = './SRC/SIM/Defaults/Weather/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3.epw'


pv_config = {
    "capacity W": 5000, # only used
    "efficiency": 0.95, # only used
    "area per W":1, # pending
    "tilt": 20, # pending
    "azimuth": 0, # pending
}