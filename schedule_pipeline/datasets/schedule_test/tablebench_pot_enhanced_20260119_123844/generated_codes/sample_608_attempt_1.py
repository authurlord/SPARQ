import pandas as pd

df = pd.read_csv('table.csv')
# Extract COP values at 35°C for both heat pumps
ashp_cop = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]
gshp_cop = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']['35 °C (e.g. heated screed floor)'].values[0]

# Convert to float and calculate difference
difference = float(ashp_cop) - float(gshp_cop)
print(f"Final Answer: {difference:.1f}")