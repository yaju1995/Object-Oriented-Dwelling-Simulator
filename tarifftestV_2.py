from datetime import datetime, timedelta

from SRC.SIM.Tariff.tariffHandler_V_2 import tariffHandler

th = tariffHandler(type=2, tariff_resolution=timedelta(minutes=30))

# Upload historic import tariff
th.upload_historic_tariff('./SRC/SIM/Defaults/Tariff/Irish_2026_Wholesale_price.csv')

# If you want same file for feed tariff too
th.upload_historic_feed_tariff('./SRC/SIM/Defaults/Tariff/Irish_2026_Wholesale_price.csv')

# Set current day + next day based on simulation time
th.prepare_day_ahead_tariffs(datetime(2026, 1, 1, 14, 0))

print(th.tariff.head())

# Current tariff at this timestamp
tariff, feed_tariff = th.get_tariff(datetime(2026, 1, 1, 0, 0))
print(tariff, feed_tariff)

# Next horizon tariff dataframe
df = th.get_tariff_range_df(datetime(2026, 1, 5, 14, 30))
print(df.head())
