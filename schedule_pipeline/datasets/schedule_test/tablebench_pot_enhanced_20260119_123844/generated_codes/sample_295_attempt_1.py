import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for years between 1975 and 1982
filtered_df = df[(df['year'] >= '1975') & (df['year'] <= '1982')]

# Initialize a dictionary to count titles
title_count = {}

# Process men's singles
for index, row in filtered_df.iterrows():
    player = row["men 's singles"]
    if player != 'no competition' and player != 'no competition':
        title_count[player] = title_count.get(player, 0) + 1

# Process men's doubles
for index, row in filtered_df.iterrows():
    players = row["men 's doubles"]
    if players != 'no competition' and players != 'no competition':
        for p in players.split():
            # Clean player name (remove any extra spaces or non-alphabetic chars)
            p_clean = p.strip()
            if p_clean:
                title_count[p_clean] = title_count.get(p_clean, 0) + 1

# Find the player with the maximum combined titles
most_titles_player = max(title_count, key=title_count.get)

print(f"Final Answer: {most_titles_player}")