import pandas as pd

df = pd.read_csv('table.csv')
# Extract COP values at 35°C for both heat pumps
ashp_35 = float(df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']['35 °C'].values[0])
gshp_35 = float(df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C'].values[0])
# Calculate difference
difference = ashp_35 - gshp_35
print(f"Final Answer: {difference:.1f}")