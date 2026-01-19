import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Triple jump events
triple_jump_data = df[df['Event'] == 'Triple jump']

# Extract positions for Asian Championships and World Championships
asian_champs_row = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs_row = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Get the positions
asian_position = int(asian_champs_row['Position'].values[0])
world_position = int(world_champs_row['Position'].values[0])

# Compare positions (lower number is better)
if asian_position < world_position:
    final_year = asian_champs_row['Year'].values[0]
else:
    final_year = None

print(f"Final Answer: {final_year}")