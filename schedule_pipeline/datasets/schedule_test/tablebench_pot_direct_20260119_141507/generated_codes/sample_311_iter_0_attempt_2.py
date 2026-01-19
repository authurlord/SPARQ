import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant columns
men_singles = df["men 's singles"]
men_doubles = df["men 's doubles"]

# Function to extract individual names from a doubles pair (split by space)
def get_doubles_names(doubles):
    if pd.isna(doubles) or doubles == '':
        return []
    names = [name.strip() for name in doubles.split(' ') if name.strip()]
    return names

# Dictionary to store matches of singles winners in doubles teams
matches = []

for idx, row in df.iterrows():
    singles_winner = row["men 's singles"]
    doubles_team = row["men 's doubles"]
    doubles_names = get_doubles_names(doubles_team)
    
    # Check if singles winner is in the doubles team
    if singles_winner in doubles_names:
        matches.append({
            "year": row["year"],
            "player": singles_winner
        })

# If there's a match, output the first (only) one
if matches:
    final_match = matches[0]
    print(f"Final Answer: {final_match['player']}, {final_match['year']}")
else:
    print("Final Answer: none")