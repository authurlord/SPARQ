import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row) since it contains labels, not data
data_rows = df.iloc[1:].copy()

# Convert the relevant columns to numeric (handle any non-numeric entries)
data_rows['Miloš Zeman'] = pd.to_numeric(data_rows['Miloš Zeman'], errors='coerce')
data_rows['Miloš Zeman_1'] = pd.to_numeric(data_rows['Miloš Zeman_1'], errors='coerce')

# Clean data: replace '-' with NaN and drop rows where either value is NaN
data_rows = data_rows.dropna(subset=['Miloš Zeman', 'Miloš Zeman_1'])

# Calculate total votes per round for Miloš Zeman
# Total votes = deputies + senators
data_rows['total_zeman_votes'] = data_rows['Miloš Zeman'] + data_rows['Miloš Zeman_1']

# Total votes across all candidates in the round (from all columns)
# We need to sum deputies and senators across all candidates in each round
# But we only have data for Václav Klaus, Jaroslava Moserová, and Miloš Zeman
# So we will sum the deputies and senators for these three candidates

# Extract deputies and senators columns for all candidates
deputy_cols = ['Václav Klaus', 'Jaroslava Moserová', 'Miloš Zeman']
senator_cols = ['Václav Klaus_1', 'Jaroslava Moserová_1', 'Miloš Zeman_1']

# Sum deputies and senators per round
data_rows['total_deputies'] = data_rows[deputy_cols].sum(axis=1)
data_rows['total_senators'] = data_rows[senator_cols].sum(axis=1)
data_rows['total_votes'] = data_rows['total_deputies'] + data_rows['total_senators']

# Check for majority win: zeman_votes > 0.5 * total_votes
data_rows['majority_win'] = (data_rows['total_zeman_votes'] > 0.5 * data_rows['total_votes'])

# Find the first round where Miloš Zeman achieved a majority win
first_majority_round = data_rows[data_rows['majority_win']].iloc[0]['Round'] if not data_rows[data_rows['majority_win']].empty else None

print(f"Final Answer: {first_majority_round}")