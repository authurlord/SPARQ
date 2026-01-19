import pandas as pd

df = pd.read_csv('table.csv')
# Extract COP values at 35°C for the two specified pumps
ashp_35c = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]
gshp_35c = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]

# Calculate difference
difference = float(ashp_35c) - float(gshp_35c)
print(f"Final Answer: {difference:.1f}")