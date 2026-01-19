import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'wins', 'second', 'third' to numeric, treating '-' as 0
df['wins'] = pd.to_numeric(df['wins'], errors='coerce').fillna(0)
df['second'] = pd.to_numeric(df['second'], errors='coerce').fillna(0)
df['third'] = pd.to_numeric(df['third'], errors='coerce').fillna(0)

# Calculate total podium finishes for each driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Group by driver and sum podium finishes
podium_totals = df.groupby('driver')['podium_finishes'].sum()

# Find the driver with the highest podium finishes
most_podium_driver = podium_totals.idxmax()

print(f"Final Answer: {most_podium_driver}")