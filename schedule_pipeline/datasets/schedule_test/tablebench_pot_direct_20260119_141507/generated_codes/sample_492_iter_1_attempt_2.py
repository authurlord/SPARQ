import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (it's a header with labels)
# The first row is ['Round', 'Václav Klaus', ...], so we skip it
data_rows = df.iloc[1:].copy()

# Define the columns for Miloš Zeman
mz_deputies_col = 'Miloš Zeman'
mz_senators_col = 'Miloš Zeman_1'

# Initialize a list to store total votes per round
rounds = []
total_votes_per_round = []

# Iterate through each row (each round)
for index, row in data_rows.iterrows():
    round_num = row['Round']
    
    # Extract values for Miloš Zeman
    deputies = row[mz_deputies_col]
    senators = row[mz_senators_col]
    
    # Convert to numeric, handling missing values (convert '-' to NaN, then to 0)
    deputies = pd.to_numeric(deputies, errors='coerce')
    senators = pd.to_numeric(senators, errors='coerce')
    
    # If either is NaN, skip this round (no valid data)
    if pd.isna(deputies) or pd.isna(senators):
        continue
    
    total_votes = deputies + senators
    mz_votes = deputies + senators  # same as total_votes
    
    # Check if majority: more than 50%
    if mz_votes > 0.5 * total_votes:
        rounds.append(round_num)
        total_votes_per_round.append(total_votes)

# If we found any round with majority, return the first one
if rounds:
    final_round = rounds[0]
    print(f"Final Answer: {final_round}")
else:
    print("Final Answer: none")