import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Find the rows for the two pump types
ashp_row = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']
gs_hp_row = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']

# Extract the 35 °C values (column index 3, since columns are ordered)
# The column '35 °C (e.g. heated screed floor)' is at index 3
ashp_35 = ashp_row.iloc[0, 3] if not ashp_row.empty else None
gs_hp_35 = gs_hp_row.iloc[0, 3] if not gs_hp_row.empty else None

# Convert to float and compute difference
if ashp_35 is not None and gs_hp_35 is not None:
    diff = float(gs_hp_35) - float(ashp_35)
    print(f"Final Answer: {diff:.1f}")
else:
    print("Final Answer: NaN")