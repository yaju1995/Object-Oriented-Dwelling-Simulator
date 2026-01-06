from datetime import datetime, timedelta
from SRC.SIM.Weather.epwHandler import EPWWeatherHandler
handler = EPWWeatherHandler("./Data/Weather/IRL_Dublin.039690_IWEC.epw")

start = datetime(2018, 3, 1)
duration = 2  # days

data = handler.get_simulation_data(start, timedelta(days=duration), pv_capacity_kw=0.0,sim_resolution=timedelta(minutes=15))

print(data.head())
print(data.tail())
print(len(data))


# import pandas as pd
# import numpy as np
# import datetime as dt
#
# # Inputs
# start_time: dt.datetime = dt.datetime(2018, 1, 1)
# duration: dt.timedelta = dt.timedelta(days=7)
# resolution: dt.timedelta = dt.timedelta(minutes=15)  # example resolution
#
# # Create datetime index
# end_time = start_time + duration
# time_index = pd.date_range(start=start_time, end=end_time, freq=resolution)
#
# # Build base DataFrame with datetime index
# df = pd.DataFrame(index=time_index)
#
# print(df.head())
# print(df.tail())
#
# # Create a second DataFrame with random demand data
# demand_df = pd.DataFrame({
#     "Demand": np.random.randint(0, 1001, size=len(time_index))
# }, index=time_index)
#
# # Merge demand data into the original DataFrame
# df = df.join(demand_df)
#
# print(df.head())
# print(df.tail())
#
# generation_df = pd.DataFrame({
#     "generation": np.random.randint(0, 1001, size=len(time_index))
# }, index=time_index)
#
#
# # Merge demand data into the original DataFrame
# df = df.join(generation_df)
#
# print(df.head())
# print(df.tail())


