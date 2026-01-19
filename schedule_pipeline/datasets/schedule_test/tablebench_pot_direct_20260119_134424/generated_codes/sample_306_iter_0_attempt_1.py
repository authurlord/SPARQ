import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where the event is 'Triple jump'
triple_jump_data = df[df['Event'] == 'Triple jump']

# Find the row for Asian Championships and World Championships
asian_champs_row = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs_row = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Extract positions
asian_position = int(asian_champs_row['Position'].iloc[0])
world_position = int(world_champs_row['Position'].iloc[0])

# Compare positions and determine the year
if asian_position < world_position:
    year = asian_champs_row['Year'].iloc[0]
else:
    year = None

print(f"Final Answer: {year}")