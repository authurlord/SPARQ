import pandas as pd

df = pd.read_csv('table.csv')

# Remove the overall row
df = df[df['driver'] != 'overall 1993 - 2012']

# Convert 'wins', 'second', 'third' to numeric, treating '-' as 0
df[['wins', 'second', 'third']] = df[['wins', 'second', 'third']].replace('-', 0).astype(int)

# Calculate total podium finishes for each driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Group by driver and sum podium finishes
podium_totals = df.groupby('driver')['podium_finishes'].sum()

# Find the driver with the maximum podium finishes
max_podium_driver = podium_totals.idxmax()

print(f"Final Answer: {max_podium_driver}")