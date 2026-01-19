import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row) since it contains labels
# The actual data starts from the second row
data_rows = df.iloc[1:].copy()

# Define the columns for Miloš Zeman
mz_deputies_col = 'Miloš Zeman'
mz_senators_col = 'Miloš Zeman_1'

# Extract the relevant columns
mz_deputies = data_rows[mz_deputies_col]
mz_senators = data_rows[mz_senators_col]

# Convert to numeric, replacing '-' with NaN
mz_deputies = pd.to_numeric(mz_deputies, errors='coerce')
mz_senators = pd.to_numeric(mz_senators, errors='coerce')

# Calculate total votes for Miloš Zeman per round
mz_total = mz_deputies + mz_senators

# Get total votes in each round (sum of all deputies and senators across all candidates)
# We need to sum over all columns (excluding headers)
# First, identify all columns that represent votes
all_vote_cols = ['Václav Klaus', 'Václav Klaus_1', 'Jaroslava Moserová', 'Jaroslava Moserová_1', 'Miloš Zeman', 'Miloš Zeman_1']

# Extract the vote data (skip first row)
vote_data = data_rows.drop(columns=['Round']).copy()

# Convert all vote columns to numeric
for col in vote_data.columns:
    vote_data[col] = pd.to_numeric(vote_data[col], errors='coerce')

# Total votes per round = sum of all votes (deputies + senators)
total_votes_per_round = vote_data.sum(axis=1)

# Now, for each round, compute percentage of Miloš Zeman's votes
mz_percentage = (mz_total / total_votes_per_round)

# Find the first round where Miloš Zeman has a majority (>50%)
# We need to ensure no division by zero
valid_rounds = mz_percentage.notna() & (total_votes_per_round != 0)

# Filter only valid rounds
mz_percentage_valid = mz_percentage[valid_rounds]
total_votes_per_round_valid = total_votes_per_round[valid_rounds]

# Check for majority win (>50%)
majority_win = mz_percentage_valid > 0.5

# Find the first round (by index) where majority win occurs
round_indices = data_rows.index[valid_rounds]
first_majority_round_idx = None
for idx, val in zip(round_indices, majority_win):
    if val:
        first_majority_round_idx = idx
        break

# Map index to round name
round_names = data_rows['Round'].values
first_majority_round = round_names[first_majority_round_idx] if first_majority_round_idx is not None else None

print(f"Final Answer: {first_majority_round}")