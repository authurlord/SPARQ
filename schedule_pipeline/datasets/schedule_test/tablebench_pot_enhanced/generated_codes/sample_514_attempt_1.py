import pandas as pd

df = pd.read_csv('table.csv')
# Remove the overall row
df = df[df['driver'] != 'overall 1993 - 2012']

# Convert 'wins', 'second', 'third' to numeric, coercing errors to NaN
df[['wins', 'second', 'third']] = df[['wins', 'second', 'third']].apply(pd.to_numeric, errors='coerce')

# Calculate total podium finishes per driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Find the driver with the maximum podium finishes
top_driver = df.loc[df['podium_finishes'].idxmax(), 'driver']

print(f"Final Answer: {top_driver}")