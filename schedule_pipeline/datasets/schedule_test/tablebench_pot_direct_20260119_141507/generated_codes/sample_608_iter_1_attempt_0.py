import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for the specific pump types
as_hp_row = df[df['Pump type and source'] == 'High efficiency ASHP, air at 0 °C']
gs_hp_row = df[df['Pump type and source'] == 'Ground source heat pump (GSHP), water at 0 °C']

# Extract 35°C performance values
as_hp_35 = as_hp_row['35 °C (e.g. heated screed floor)'].values[0] if not as_hp_row.empty else None
gs_hp_35 = gs_hp_row['35 °C (e.g. heated screed floor)'].values[0] if not gs_hp_row.empty else None

# Convert to float and compute difference
if as_hp_35 is not None and gs_hp_35 is not None:
    difference = float(gs_hp_35) - float(as_hp_35)
    print(f"Final Answer: {difference}")
else:
    print("Final Answer: NaN")