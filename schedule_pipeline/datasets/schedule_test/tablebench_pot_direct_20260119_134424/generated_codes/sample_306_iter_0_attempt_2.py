import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the event is 'Triple jump'
triple_jump_data = df[df['Event'] == 'Triple jump']

# Find the year of Asian Championships and World Championships
asian_champs_row = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs_row = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Extract the position and year for both events
asian_position = asian_champs_row['Position'].values[0]
asian_year = asian_champs_row['Year'].values[0]
world_position = world_champs_row['Position'].values[0]
world_year = world_champs_row['Year'].values[0]

# Compare positions: lower number means better position
if asian_position < world_position:
    print(f"Final Answer: {asian_year}")
else:
    print(f"Final Answer: None")