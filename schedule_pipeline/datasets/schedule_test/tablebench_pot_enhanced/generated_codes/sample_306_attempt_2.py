import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Triple Jump events
triple_jump_data = df[df['Event'] == 'Triple jump']

# Extract the position for Asian Championships and World Championships
asian_championships = triple_jump_data[triple_jump_data['Competition'] == 'Asian Championships']
world_championships = triple_jump_data[triple_jump_data['Competition'] == 'World Championships']

# Get the year where Asian Championships position is better (lower number) than World Championships
if not asian_championships.empty and not world_championships.empty:
    asian_year = asian_championships.iloc[0]['Year']
    world_year = world_championships.iloc[0]['Year']
    asian_pos = int(asian_championships.iloc[0]['Position'])
    world_pos = int(world_championships.iloc[0]['Position'])

    if asian_pos < world_pos:
        print(f"Final Answer: {asian_year}")
    else:
        print("Final Answer: None")
else:
    print("Final Answer: None")