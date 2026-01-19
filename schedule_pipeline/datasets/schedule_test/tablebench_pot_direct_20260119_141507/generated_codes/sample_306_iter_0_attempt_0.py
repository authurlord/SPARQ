import pandas as pd

df = pd.read_csv('table.csv')

# Filter for Triple jump events
df_trip = df[df['Event'] == 'Triple jump']

# Extract positions and years for Asian Championships and World Championships
asian_champs = df_trip[df_trip['Competition'] == 'Asian Championships']
world_champs = df_trip[df_trip['Competition'] == 'World Championships']

# Convert 'Position' to numeric (e.g., '21st (q)' -> 21)
def extract_position(pos):
    try:
        return int(pos.split('th')[0])
    except:
        return float('inf')

asian_pos = extract_position(asian_champs['Position'].iloc[0]) if not asian_champs.empty else float('inf')
world_pos = extract_position(world_champs['Position'].iloc[0]) if not world_champs.empty else float('inf')

# Check if Asian position is better (smaller number)
if asian_pos < world_pos:
    year = asian_champs['Year'].iloc[0]
else:
    year = None

print(f"Final Answer: {year}")