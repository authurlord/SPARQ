import pandas as pd

df = pd.read_csv('table.csv')

# Filter for World Championships and 800m events
world_championships_800m = df[(df['Competition'] == 'World Championships') & 
                             (df['Event'].str.contains('800 m|800 metres', case=False))]

# Remove DNF entries and convert positions to numeric for comparison
world_championships_800m = world_championships_800m[world_championships_800m['Position'] != 'DNF']
world_championships_800m['Position'] = pd.to_numeric(world_championships_800m['Position'], errors='coerce')

# Drop rows with invalid positions
world_championships_800m = world_championships_800m.dropna(subset=['Position'])

# Find the best (lowest) position
best_position = world_championships_800m['Position'].min()

# Get the year corresponding to the best position
best_year = world_championships_800m[world_championships_800m['Position'] == best_position]['Year'].iloc[0]

print(f"Final Answer: {best_year}")