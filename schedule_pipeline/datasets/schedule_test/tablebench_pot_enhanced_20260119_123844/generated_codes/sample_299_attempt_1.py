import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships and 800m events
world_championships_800m = df[(df['Competition'] == 'World Championships') & 
                             (df['Event'].str.contains('800 m|800 metres', case=False))]

# Find the row with the best (lowest) position
best_position_row = world_championships_800m.loc[world_championships_800m['Position'].str.replace('th', '').str.replace('nd', '').str.replace('rd', '').str.replace('st', '').astype(int).idxmin()]

# Extract the year
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")