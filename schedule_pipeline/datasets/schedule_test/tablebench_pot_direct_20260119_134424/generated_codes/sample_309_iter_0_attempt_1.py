import pandas as pd

df = pd.read_csv('table.csv')

# Extract women's singles champions and their years
women_singles = df['womens singles'].tolist()

# Create a list of women who won mixed doubles (extract first name from each pair)
mixed_doubles_winners = []
for md in df['mixed doubles']:
    if md != 'no competition':
        # Split by space and extract the first name (assuming first name is the winner)
        names = md.split()
        # Only consider the first name (assumption based on data pattern)
        mixed_doubles_winners.append(names[0])

# Count how many times each women's singles champion also won mixed doubles
winner_count = {}
for ws in women_singles:
    if ws != 'no competition':
        # Split name to get first name (last name might vary)
        ws_name = ws.split()[0]
        if ws_name in mixed_doubles_winners:
            winner_count[ws_name] = winner_count.get(ws_name, 0) + 1

# Find the woman with the most titles among those who also won mixed doubles
if winner_count:
    max_wins = max(winner_count.values())
    top_champion = [name for name, count in winner_count.items() if count == max_wins][0]
else:
    top_champion = "None"

print(f"Final Answer: {top_champion}")