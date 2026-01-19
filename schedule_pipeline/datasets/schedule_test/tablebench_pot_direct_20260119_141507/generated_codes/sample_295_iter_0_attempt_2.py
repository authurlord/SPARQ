import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows for years between 1975 and 1982 (inclusive)
# We need to parse the year column and keep only those where year is in 1975-1982
filtered_rows = []
for index, row in df.iterrows():
    year_str = row['year']
    # Check if year is a single year and within 1975-1982
    if isinstance(year_str, str):
        # Handle cases like '1975', '1978', '1990 1997' — split and check
        year_parts = year_str.split()
        if len(year_parts) == 1:
            year = int(year_parts[0])
            if 1975 <= year <= 1982:
                filtered_rows.append(row)
        elif len(year_parts) > 1:
            # e.g., '1990 1997' — skip if any part is outside 1975-1982
            years = [int(y) for y in year_parts]
            if any(1975 <= y <= 1982 for y in years):
                filtered_rows.append(row)

# Create a list of players from men's singles and men's doubles
player_count = {}

# Process men's singles
for row in filtered_rows:
    men_singles = row["men 's singles"]
    if men_singles != "no competition":
        # Split if there are multiple names
        if ',' in men_singles:
            names = [name.strip() for name in men_singles.split(',')]
        else:
            names = [men_singles.strip()]
        for name in names:
            player_count[name] = player_count.get(name, 0) + 1

# Process men's doubles
for row in filtered_rows:
    men_doubles = row["men 's doubles"]
    if men_doubles != "no competition":
        if ',' in men_doubles:
            names = [name.strip() for name in men_doubles.split(',')]
        else:
            names = [men_doubles.strip()]
        for name in names:
            player_count[name] = player_count.get(name, 0) + 1

# Find the player with the maximum total
if player_count:
    winner = max(player_count, key=player_count.get)
    print(f"Final Answer: {winner}")
else:
    print("Final Answer: no player")