import pandas as pd
import re

df = pd.read_csv('table.csv')

def extract_player(name):
    # Remove any "(X/Y)" or similar parts
    match = re.search(r'^(.+?)(?:\s*\([^)]*\))?$', name)
    return match.group(1) if match else name

# Create a list to store years where same player won both titles
matching_years = []

for index, row in df.iterrows():
    senior_pga = row['senior pga championship']
    senior_players = row['senior players championship']
    
    # Skip rows where either event is "not held", "founded in 1989", or missing
    if pd.isna(senior_pga) or pd.isna(senior_players):
        continue
    
    pga_player = extract_player(senior_pga)
    players_player = extract_player(senior_players)
    
    # Check if the player names match
    if pga_player == players_player:
        matching_years.append(row['year'])

# If there are matches, print the first one (or all if multiple)
if matching_years:
    print(f"Final Answer: {matching_years[0]}")
else:
    print("Final Answer: none")