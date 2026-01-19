import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Triple jump events
triple_jump_data = df[df['Event'] == 'Triple jump']

# Extract positions for Asian Championships and World Championships
asian_champs_pos = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']['Position'].iloc[0]
world_champs_pos = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']['Position'].iloc[0]

# Convert positions to numeric for comparison (e.g., 4th -> 4, 21st -> 21)
asian_pos_num = int(asian_champs_pos.replace('th', '').replace('rd', '').replace('nd', '').replace('st', ''))
world_pos_num = int(world_champs_pos.replace('th', '').replace('rd', '').replace('nd', '').replace('st', ''))

# Find the year when Asian Championships position is better (lower number)
if asian_pos_num < world_pos_num:
    year = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']['Year'].iloc[0]
    print(f"Final Answer: {year}")
else:
    print("Final Answer: None")