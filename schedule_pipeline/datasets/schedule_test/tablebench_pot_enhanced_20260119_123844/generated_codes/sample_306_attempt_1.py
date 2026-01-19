import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where the event is 'Triple jump'
triple_jump_data = df[df['Event'] == 'Triple jump']

# Extract data for Asian Championships and World Championships
asian_champs = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Merge on Year to compare positions
comparison = pd.merge(asian_champs[['Year', 'Position']], world_champs[['Year', 'Position']], on='Year', suffixes=('_asian', '_world'))

# Find the year where position at Asian Championships is better (lower number) than at World Championships
better_year = comparison[comparison['Position_asian'] < comparison['Position_world']]['Year'].iloc[0]

print(f"Final Answer: {better_year}")