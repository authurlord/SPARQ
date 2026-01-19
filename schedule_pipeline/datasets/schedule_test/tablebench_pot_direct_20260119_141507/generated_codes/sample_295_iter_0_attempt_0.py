import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter years between 1975 and 1982 (inclusive)
valid_years = []
for row in df.values:
    year = row[0]
    if isinstance(year, str):
        # Check if year is a single year within 1975–1982
        if year.isdigit() and 1975 <= int(year) <= 1982:
            valid_years.append(row)
    else:
        continue

# Extract only valid rows
filtered_df = pd.DataFrame(valid_years, columns=df.columns)

# Clean up men's singles and men's doubles columns
def extract_players(winners_str):
    if winners_str == 'no competition':
        return []
    if isinstance(winners_str, str):
        # Split by space or comma and filter names
        names = [name.strip() for name in winners_str.split() if name.strip()]
        return names
    return []

# Create a dictionary to count titles
title_count = {}

# Process men's singles
for _, row in filtered_df.iterrows():
    singles = row["men 's singles"]
    if singles != 'no competition':
        players = extract_players(singles)
        for p in players:
            title_count[p] = title_count.get(p, 0) + 1

# Process men's doubles
for _, row in filtered_df.iterrows():
    doubles = row["men 's doubles"]
    if doubles != 'no competition':
        players = extract_players(doubles)
        for p in players:
            title_count[p] = title_count.get(p, 0) + 1

# Find the player with the most titles
if title_count:
    winner = max(title_count, key=title_count.get)
    print(f"Final Answer: {winner}")
else:
    print("Final Answer: none")