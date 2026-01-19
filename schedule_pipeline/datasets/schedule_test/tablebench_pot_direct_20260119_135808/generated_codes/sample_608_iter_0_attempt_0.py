import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for the two heat pumps at 35°C
ashp_row = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']
gshp_row = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']

# Extract COP values at 35°C (column index 3)
ashp_cop_35 = float(ashp_row.iloc[0]['35 °C (e.g. heated screed floor)'])
gshp_cop_35 = float(gshp_row.iloc[0]['35 °C (e.g. heated screed floor)'])

# Calculate difference
difference = gshp_cop_35 - ashp_cop_35
print(f"Final Answer: {difference:.1f}")