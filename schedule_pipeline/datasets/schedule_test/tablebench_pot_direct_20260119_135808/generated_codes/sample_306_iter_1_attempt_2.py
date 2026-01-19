import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Event is 'Triple jump'
triple_jump_data = df[df['Event'] == 'Triple jump']

# Function to convert position like '4th' to integer
def parse_position(pos):
    return int(''.join(filter(str.isdigit, pos)))

# Apply parsing to Position column
triple_jump_data['Position_num'] = triple_jump_data['Position'].apply(parse_position)

# Separate data for Asian Championships and World Championships
asian_champs = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Merge on Year to compare positions
merged_data = pd.merge(asian_champs[['Year', 'Position_num']], 
                       world_champs[['Year', 'Position_num']], 
                       on='Year', 
                       suffixes=('_asian', '_world'))

# Find year where Asian Championships position is better (lower number)
better_year = merged_data[merged_data['Position_num_asian'] < merged_data['Position_num_world']]['Year'].iloc[0]

print(f"Final Answer: {better_year}")