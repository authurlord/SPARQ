import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Triple jump events
triple_jump_data = df[df['Event'] == 'Triple jump']

# Function to convert position string (e.g., '4th', '3rd') to integer
def extract_position(pos):
    return int(''.join(filter(str.isdigit, pos)))

# Extract positions for Asian Championships and World Championships
asian_champs = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Extract the year and position for each
asian_year = asian_champs.iloc[0]['Year']
asian_pos = extract_position(asian_champs.iloc[0]['Position'])

world_year = world_champs.iloc[0]['Year']
world_pos = extract_position(world_champs.iloc[0]['Position'])

# Compare positions
if asian_pos < world_pos:
    print(f"Final Answer: {asian_year}")
else:
    print("Final Answer: None")