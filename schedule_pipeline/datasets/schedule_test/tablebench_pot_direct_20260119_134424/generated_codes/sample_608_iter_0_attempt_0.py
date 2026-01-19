import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for the two heat pumps
ashp_row = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']
gshp_row = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']

# Extract COP values at 35°C (index 2)
ashp_cop_35 = float(ashp_row.iloc[0, 2])
gshp_cop_35 = float(gshp_row.iloc[0, 2])

# Calculate difference
difference = ashp_cop_35 - gshp_cop_35
print(f"Final Answer: {difference:.1f}")