import pandas as pd

df = pd.read_csv('table.csv')

# Extract women's singles champions and mixed doubles champions
women_singles = df['womens singles'].str.split(',').str[0].str.strip()
mixed_doubles = df['mixed doubles'].str.split(',').str[1].str.strip()

# Find women who are both women's singles champions and mixed doubles champions
singles_champions = set(women_singles)
mixed_doubles_champions = set(mixed_doubles)

# Find intersection (champions who won both)
both_winners = singles_champions.intersection(mixed_doubles_champions)

# Count how many times each woman in the intersection won women's singles
title_count = {}
for winner in women_singles:
    if winner in both_winners:
        title_count[winner] = title_count.get(winner, 0) + 1

# Find the one with the most titles
most_titles = max(title_count, key=title_count.get)

print(f"Final Answer: {most_titles}")