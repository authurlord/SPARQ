import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Triple jump events at Asian Championships and World Championships
asian_champs = df[(df['Event'] == 'Triple jump') & (df['Competition'] == 'Asian Championships')]
world_champs = df[(df['Event'] == 'Triple jump') & (df['Competition'] == 'World Championships')]

# Extract positions and years
asian_pos = asian_champs['Position'].values[0]
world_pos = world_champs['Position'].values[0]
asian_year = asian_champs['Year'].values[0]
world_year = world_champs['Year'].values[0]

# Compare positions: lower number means better position
if int(asian_pos) < int(world_pos.replace('th', '').replace('nd', '').replace('rd', '').replace('st', '')):
    print(f"Final Answer: {asian_year}")
else:
    print(f"Final Answer: None")