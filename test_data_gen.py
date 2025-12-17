import matplotlib.pyplot as plt

from SRC.SIM.data_generators import PatternGenerationHandler
from datetime import datetime, timedelta
from matplotlib.pyplot import plot

handler = PatternGenerationHandler(
    model_name="normal",
    csv_path="./SRC/SIM/Example/Demand/15_min_normal.csv"
)

print(handler.df.head())
print(handler.df.tail())
handler.visualize_csv_pattern()

df = handler.get_simulation_data(
    start_date=datetime(2025, 1, 1),
    duration_days=timedelta(days=5),
    resolution=timedelta(minutes=15),
    column_name='Demand',
    seed=7
)

print(df.head())
print(df.tail())

# df['Demand'].plot(
#     kind='line',
#     # marker='o',
#     title='Line Plot of Value Column',
#     grid=True
# )
# plt.show()




