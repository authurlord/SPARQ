import pandas as pd

df = pd.read_csv('table.csv')

# Find the values at 35 °C for the two specified pump types
as_hp_35c = df[df['Pump type and source'] == 'High-efficiency ASHP, air at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]
gs_hp_35c = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]

difference = gs_hp_35c - as_hp_35c
print(f"Final Answer: {difference}")