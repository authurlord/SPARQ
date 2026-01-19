import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Create a list of years and check for matches
matches = []

for idx, row in df.iterrows():
    year = row['year']
    men_singles = row["men 's singles"]
    men_doubles = row["men 's doubles"]
    
    # Split doubles into individual names
    doubles_names = men_doubles.split()
    
    # Check if the men's singles winner is in the doubles pair
    if men_singles in doubles_names:
        matches.append((men_singles, year))

# If there's a match, return it; otherwise, no one has both
if matches:
    final_player, final_year = matches[0]
    print(f"Final Answer: {final_player}, {final_year}")
else:
    print("Final Answer: none, none")