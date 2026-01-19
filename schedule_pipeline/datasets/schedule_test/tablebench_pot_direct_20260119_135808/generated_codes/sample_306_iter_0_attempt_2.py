import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Asian Championships and World Championships in Triple Jump
asian_champs = df[(df['Competition'] == 'Asian Championships') & (df['Event'] == 'Triple jump')]
world_champs = df[(df['Competition'] == 'World Championships') & (df['Event'] == 'Triple jump')]

# Extract the year and position for both
asian_year = asian_champs['Year'].values[0]
asian_position = int(asian_champs['Position'].values[0])

world_year = world_champs['Year'].values[0]
world_position = int(world_champs['Position'].values[0])

# Compare positions (lower number is better)
if asian_position < world_position:
    print(f"Final Answer: {asian_year}")
else:
    print(f"Final Answer: None")