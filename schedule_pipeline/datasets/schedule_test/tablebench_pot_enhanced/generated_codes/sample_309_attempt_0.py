import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract women's singles champions and mixed doubles champions
women_singles = df['womens singles'].tolist()
mixed_doubles = df['mixed doubles'].tolist()

# Create a dictionary to count how many times each women's singles champion has won mixed doubles
title_count = {}

for i, winner in enumerate(women_singles):
    # Clean the name (remove city info if present)
    clean_winner = winner.split(',')[0].strip()
    # Check if this champion appears in mixed doubles
    for md_winner in mixed_doubles:
        if isinstance(md_winner, str) and clean_winner in md_winner:
            title_count[clean_winner] = title_count.get(clean_winner, 0) + 1

# Find the champion with the most titles (max count)
if title_count:
    max_champion = max(title_count, key=title_count.get)
    print(f"Final Answer: {max_champion}")
else:
    print("Final Answer: None")