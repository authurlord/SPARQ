import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to integers (handle missing '-' entries)
def parse_podium(value):
    if value == '-' or value == '':
        return 0
    return int(value)

# Apply parsing to wins, second, third
df['wins'] = df['wins'].apply(parse_podium)
df['second'] = df['second'].apply(parse_podium)
df['third'] = df['third'].apply(parse_podium)

# Calculate total podium finishes per driver
df['podium_finishes'] = df['wins'] + df['second'] + df['third']

# Group by driver and find the one with maximum podium finishes
driver_podiums = df.groupby('driver')['podium_finishes'].sum()
max_podium_driver = driver_podiums.idxmax()

print(f"Final Answer: {max_podium_driver}")