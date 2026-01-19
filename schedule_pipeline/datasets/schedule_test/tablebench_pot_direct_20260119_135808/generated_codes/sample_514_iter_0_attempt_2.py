import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'wins', 'second', and 'third' to numeric, treating '-' as 0
df['wins'] = pd.to_numeric(df['wins'], errors='coerce').fillna(0)
df['second'] = pd.to_numeric(df['second'], errors='coerce').fillna(0)
df['third'] = pd.to_numeric(df['third'], errors='coerce').fillna(0)

# Calculate total podium finishes for each driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Find the driver with the maximum podium finishes
max_podium_driver = df.loc[df['podium_finishes'].idxmax(), 'driver']

print(f"Final Answer: {max_podium_driver}")