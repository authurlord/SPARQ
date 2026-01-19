import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Event is 'Triple jump'
triple_jump_data = df[df['Event'] == 'Triple jump']

# Clean the Position column to extract numeric values
triple_jump_data['Position'] = triple_jump_data['Position'].str.replace(r'[a-zA-Z]', '', regex=True).astype(int)

# Filter for Asian Championships and World Championships
asian_championships = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_championships = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Merge the two datasets on Year to compare
comparison = pd.merge(asian_championships[['Year', 'Position']], 
                      world_championships[['Year', 'Position']], 
                      on='Year', suffixes=('_asian', '_world'))

# Find the year where Asian Championships position is better (lower number)
better_year = comparison[comparison['Position_asian'] < comparison['Position_world']]['Year'].iloc[0]

print(f"Final Answer: {better_year}")