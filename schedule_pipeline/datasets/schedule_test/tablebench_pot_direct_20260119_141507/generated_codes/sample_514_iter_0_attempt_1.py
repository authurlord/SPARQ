import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'wins', 'second', 'third' to numeric, treating '-' as NaN and then replacing with 0
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['second'] = pd.to_numeric(df['second'], errors='coerce')
df['third'] = pd.to_numeric(df['third'], errors='coerce')

# Sum podium finishes per driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Group by driver and sum podium finishes
podium_summary = df.groupby('driver')['podium_finishes'].sum()

# Find the driver with the maximum podium finishes
top_driver = podium_summary.idxmax()
max_podiums = podium_summary.max()

print(f"Final Answer: {top_driver}")