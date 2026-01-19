import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Event is 'Triple jump'
df_trip = df[df['Event'] == 'Triple jump']

# Filter for Asian Championships and World Championships
asian_champs = df_trip[df_trip['Competition'] == 'Asian Championships']
world_champs = df_trip[df_trip['Competition'] == 'World Championships']

# Convert Position to numeric (remove 'q' suffix if present)
def parse_position(pos):
    if isinstance(pos, str) and 'q' in pos:
        return int(pos.split('(')[0])
    return int(pos)

asian_pos = asian_champs['Position'].apply(parse_position)
world_pos = world_champs['Position'].apply(parse_position)

# Find years where Asian position is better (smaller number) than World position
valid_years = []
for idx, row in asian_champs.iterrows():
    year = row['Year']
    asian_rank = parse_position(row['Position'])
    world_rank = None
    # Find matching world championships record in same year
    world_row = world_champs[world_champs['Year'] == year]
    if not world_row.empty:
        world_rank = parse_position(world_row.iloc[0]['Position'])
    if asian_rank < world_rank:
        valid_years.append(year)

print(f"Final Answer: {valid_years}")