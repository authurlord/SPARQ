import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for years between 1975 and 1982
filtered_df = df[(df['year'] >= '1975') & (df['year'] <= '1982')]

# Extract men's singles and men's doubles winners
men_singles_winners = filtered_df['men \'s singles'].tolist()
men_doubles_winners = filtered_df['men \'s doubles'].tolist()

# Combine and count occurrences
from collections import Counter
all_winners = []

# Process men's singles
for winner in men_singles_winners:
    if winner != 'no competition':
        all_winners.append(winner)

# Process men's doubles (split by space and take first name)
for doubles in men_doubles_winners:
    if doubles != 'no competition':
        # Split by space and take the first name
        players = doubles.split()
        all_winners.append(players[0])

# Count wins
win_count = Counter(all_winners)

# Find the player with the most wins
most_wins_player = win_count.most_common(1)[0][0]

print(f"Final Answer: {most_wins_player}")