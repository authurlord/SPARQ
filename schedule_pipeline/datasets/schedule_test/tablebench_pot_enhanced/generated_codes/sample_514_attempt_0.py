import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Replace '-' with 0 for numerical calculation
df[['wins', 'second', 'third']] = df[['wins', 'second', 'third']].replace('-', 0)

# Convert columns to integers
df['wins'] = df['wins'].astype(int)
df['second'] = df['second'].astype(int)
df['third'] = df['third'].astype(int)

# Calculate total podium finishes for each driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Group by driver and sum podium finishes
podium_totals = df.groupby('driver')['podium_finishes'].sum()

# Find the driver with the maximum podium finishes
most_podium_driver = podium_totals.idxmax()

print(f"Final Answer: {most_podium_driver}")