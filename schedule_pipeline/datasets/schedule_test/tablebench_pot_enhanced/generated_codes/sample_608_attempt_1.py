import pandas as pd

df = pd.read_csv('table.csv')
# Extract COP values at 35°C for High-efficiency ASHP and GSHP
ashp_35c = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]
gshp_35c = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]

# Convert to float and compute difference
difference = float(ashp_35c) - float(gshp_35c)
print(f"Final Answer: {difference:.1f}")