import pandas as pd

df = pd.read_csv('table.csv')

# Find the row for High-efficiency air source heat pump (ASHP) at 0 °C
ashp_row = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']
# Find the row for Ground source heat pump (GSHP) at 0 °C
gsph_row = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']

# Extract 35°C values (index 2 in columns)
ashp_35c = ashp_row.iloc[0]['35 °C (e.g. heated screed floor)']
gsph_35c = gsph_row.iloc[0]['35 °C (e.g. heated screed floor)']

# Convert to float and compute difference
ashp_value = float(ashp_35c) if pd.notna(ashp_35c) else 0
gsph_value = float(gsph_35c) if pd.notna(gsph_35c) else 0

difference = gsph_value - ashp_value
print(f"Final Answer: {difference:.1f}")