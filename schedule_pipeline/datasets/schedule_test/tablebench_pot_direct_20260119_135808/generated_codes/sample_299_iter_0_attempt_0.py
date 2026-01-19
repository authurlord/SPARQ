import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships and 800m event
world_champs_800m = df[(df['Competition'] == 'World Championships') & (df['Event'] == '800 m')]
# Find the row with the best position (lowest position number)
best_position_row = world_champs_800m.loc[world_champs_800m['Position'].str.extract('(\d+)', expand=False).astype(int).idxmin()]
# Extract the year
best_year = best_position_row['Year']
print(f"Final Answer: {best_year}")