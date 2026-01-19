import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships and 800m event
world_champs_800m = df[(df['Competition'] == 'World Championships') & (df['Event'] == '800 m')]

# Convert Position to numeric, treating DNF as a very poor result (e.g., 999)
world_champs_800m['Position'] = pd.to_numeric(world_champs_800m['Position'], errors='coerce')
world_champs_800m['Position'] = world_champs_800m['Position'].fillna(999)

# Find the row with the best (minimum) position
best_position_row = world_champs_800m.loc[world_champs_800m['Position'].idxmin()]

# Get the year of the best performance
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")