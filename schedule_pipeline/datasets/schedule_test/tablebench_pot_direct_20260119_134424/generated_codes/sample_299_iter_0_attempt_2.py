import pandas as pd

df = pd.read_csv('table.csv')
# Filter for World Championships and 800m events
world_champs_800m = df[(df['Competition'] == 'World Championships') & 
                       (df['Event'].str.contains('800 m|800 metres', case=False))]

# Remove DNF entries
valid_results = world_champs_800m[world_champs_800m['Position'] != 'DNF']

# Convert Position to numeric for comparison, handling 'sf' or 'h' suffixes
def clean_position(pos):
    try:
        return int(pos.split('(')[0]) if isinstance(pos, str) else pos
    except:
        return float('inf')

valid_results['Position_Num'] = valid_results['Position'].apply(clean_position)
best_position_row = valid_results.loc[valid_results['Position_Num'].idxmin()]

print(f"Final Answer: {best_position_row['Year']}")