import pandas as pd

df = pd.read_csv('table.csv')

# Filter for World Championships and 800m event
world_championships_800m = df[(df['Competition'] == 'World Championships') & (df['Event'] == '800 m')]

# Find the row with the best (lowest) position
best_position_row = world_championships_800m.loc[world_championships_800m['Position'].astype(str).str.replace('th', '').str.replace('rd', '').str.replace('nd', '').str.replace('st', '').astype(int).idxmin()]

# Extract the year
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")