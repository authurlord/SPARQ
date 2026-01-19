import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships and 800m events
world_champs_800m = df[(df['Competition'] == 'World Championships') & 
                       (df['Event'].str.contains('800 m|800 metres', case=False))]
# Find the best position (lowest numerical value in 'Position')
best_position_row = world_champs_800m.loc[world_champs_800m['Position'].astype(str).str.replace('sf', '').str.strip().astype(int).idxmin()]
year = best_position_row['Year']
print(f"Final Answer: {year}")