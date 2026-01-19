import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Triple jump events at Asian Championships and World Championships
triple_jump_data = df[df['Event'] == 'Triple jump']
asian_champs = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_champs = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Extract the year where Asian Championships position is better (lower number) than World Championships
asian_year = asian_champs['Year'].iloc[0]
world_year = world_champs['Year'].iloc[0]

# Compare positions
asian_position = int(asian_champs['Position'].iloc[0])
world_position = int(world_champs['Position'].iloc[0])

if asian_position < world_position:
    print(f"Final Answer: {asian_year}")
else:
    print("Final Answer: None")