import pandas as pd

df = pd.read_csv('table.csv')

# Find the values for 35°C for both ASHP and GSHP
as_hp_35c = df[df['Pump type and source'] == 'High-efficiency air source heat pump (ASHP), air at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]
gs_hp_35c = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]

difference = gs_hp_35c - as_hp_35c
print(f"Final Answer: {difference}")