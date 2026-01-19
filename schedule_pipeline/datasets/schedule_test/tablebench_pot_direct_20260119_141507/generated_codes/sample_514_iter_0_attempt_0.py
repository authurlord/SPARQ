import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert "second" and "third" columns to numeric, treating '-' as NaN
df['second'] = pd.to_numeric(df['second'], errors='coerce')
df['third'] = pd.to_numeric(df['third'], errors='coerce')

# Calculate total podium finishes per driver
df['podium_finishes'] = df['second'] + df['third']

# Group by driver and sum podium finishes
podium_summary = df.groupby('driver')['podium_finishes'].sum()

# Find the driver with the maximum podium finishes
top_driver = podium_summary.idxmax()
print(f"Final Answer: {top_driver}")