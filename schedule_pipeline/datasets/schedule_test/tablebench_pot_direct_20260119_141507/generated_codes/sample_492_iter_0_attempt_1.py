import pandas as pd

df = pd.read_csv('table.csv')

# Filter out header row
data = df.iloc[1:].copy()

# Extract Miloš Zeman's deputies and senators
milo_zeman_deputies = data['Miloš Zeman'].dropna()
milo_zeman_senators = data['Miloš Zeman_1'].dropna()

# Only consider rows where both values are present
valid_rows = data[(data['Miloš Zeman'].notna()) & (data['Miloš Zeman_1'].notna())]

# Calculate total votes per round and check if Miloš Zeman has majority
majority_rounds = []
for idx, row in valid_rows.iterrows():
    total_votes = row['Miloš Zeman'] + row['Miloš Zeman_1']
    milo_zeman_votes = row['Miloš Zeman']
    if milo_zeman_votes > total_votes * 0.5:
        majority_rounds.append(row['Round'])

# If majority rounds exist, return the first one
if majority_rounds:
    first_majority_round = majority_rounds[0]
else:
    first_majority_round = None

print(f"Final Answer: {first_majority_round}")