import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Triple jump events at Asian Championships and World Championships
asi_champs = df[(df['Event'] == 'Triple jump') & (df['Competition'] == 'Asian Championships')]
world_champs = df[(df['Event'] == 'Triple jump') & (df['Competition'] == 'World Championships')]

# Extract the years and positions
asi_year = asi_champs['Year'].values[0]
asi_pos = int(asi_champs['Position'].values[0])
world_year = world_champs['Year'].values[0]
world_pos = int(world_champs['Position'].values[0])

# Compare positions: lower number is better
if asi_pos < world_pos:
    print(f"Final Answer: {asi_year}")
else:
    print(f"Final Answer: None")