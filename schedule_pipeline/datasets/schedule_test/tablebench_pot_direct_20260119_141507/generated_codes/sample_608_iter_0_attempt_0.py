import pandas as pd

df = pd.read_csv('table.csv')

# Find the values for High-efficiency ASHP at 35 °C and GSHP at 35 °C
ashp_35c = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]
gs_hp_35c = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]

# Calculate the difference
difference = gs_hp_35c - ashp_35c
print(f"Final Answer: {difference}")