import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for years between 1975 and 1982
filtered_df = df[(df['year'] >= '1975') & (df['year'] <= '1982')]

# Combine men's singles and men's doubles winners
men_singles_winners = filtered_df["men 's singles"].tolist()
men_doubles_winners = filtered_df["men 's doubles"].str.split(' ', expand=False).str[0].tolist()

# Combine all winners
all_men_winners = men_singles_winners + men_doubles_winners

# Count occurrences
from collections import Counter
winner_counts = Counter(all_men_winners)

# Find the player with the most titles
most_wins_player = winner_counts.most_common(1)[0][0]

print(f"Final Answer: {most_wins_player}")