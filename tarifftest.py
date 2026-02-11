from SRC.SIM.Tariff.tariffHandler import tariffHandler
from datetime import datetime

tariff = tariffHandler()



tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')
tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example.csv')

value = tariff.get_tariff(datetime(year=2018,month=1,day=1,hour=1, minute=15))

print(value)
print(tariff.tariff['value'].max())
print(tariff.tariff['value'].min())

print(tariff.max_tariff)
print(tariff.min_tariff)

print(tariff.max_feed_tariff)
print(tariff.min_feed_tariff)