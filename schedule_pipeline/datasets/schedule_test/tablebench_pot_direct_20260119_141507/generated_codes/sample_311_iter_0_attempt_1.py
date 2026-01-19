import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract relevant columns
men_singles = df["men 's singles"]
men_doubles = df["men 's doubles"]

# List to store matches
matches = []

# Iterate through each row
for index, row in df.iterrows():
    year = row['year']
    ms = row['men "s singles']
    md = row['men "s doubles']
    
    # Split doubles into two names
    md_names = md.split()
    
    # Check if the men's singles name is in the doubles pair
    if ms in md_names:
        matches.append((ms, year))

# If there's exactly one match, that's our answer
if matches:
    player, year = matches[0]
    print(f"Final Answer: {player}, {year}")
else:
    print("Final Answer: none")