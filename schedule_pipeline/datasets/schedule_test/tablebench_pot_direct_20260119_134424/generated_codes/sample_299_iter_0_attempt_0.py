import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships and 800m event
world_champs_800m = df[(df['Competition'] == 'World Championships') & (df['Event'] == '800 m')]
# Find the best position (lowest numerical value)
best_position_row = world_champs_800m.loc[world_champs_800m['Position'].str.replace('th', '').astype(int).idxmin()]
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")