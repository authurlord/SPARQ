import pandas as pd

df = pd.read_csv('table.csv')
# Extract COP for High-efficiency ASHP at 35 °C (column index 3)
asph_cop_35 = float(df.loc[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C', '35 °C (e.g. heated screed floor)'].values[0])
# Extract COP for GSHP at 35 °C (column index 3)
gshp_cop_35 = float(df.loc[df['Pump type and source'] == 'GSHP, ground at 10 °C', '35 °C (e.g. heated screed floor)'].values[0])
# Calculate difference
difference = asph_cop_35 - gshp_cop_35
print(f"Final Answer: {difference:.1f}")