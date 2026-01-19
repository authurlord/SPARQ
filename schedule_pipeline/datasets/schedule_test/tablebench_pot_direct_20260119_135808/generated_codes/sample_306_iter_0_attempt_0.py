import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 'Asian Championships' and 'World Championships' in the 'Event' column
asian_champs = df[(df['Competition'] == 'Asian Championships') & (df['Event'] == 'Triple jump')]
world_champs = df[(df['Competition'] == 'World Championships') & (df['Event'] == 'Triple jump')]

# Get the year of Asian Championships and World Championships
asian_year = asian_champs['Year'].iloc[0]
world_year = world_champs['Year'].iloc[0]

# Compare positions (lower is better)
asian_position = int(asian_champs['Position'].iloc[0])
world_position = int(world_champs['Position'].iloc[0])

# Check if Asian Championships position is better (lower number) than World Championships
if asian_position < world_position:
    print(f"Final Answer: {asian_year}")
else:
    print(f"Final Answer: {world_year}")