import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'womens singles' and 'mixed doubles' columns
womens_singles = df['womens singles']
mixed_doubles = df['mixed doubles']

# Create a dictionary to count titles for women's singles champions who also won mixed doubles
title_count = {}

# Iterate through the rows
for idx, row in df.iterrows():
    womens_singles_winner = row['womens singles'].split(',')[0].strip()
    mixed_doubles_winner = row['mixed doubles'].split(',')[0].strip()
    
    # Check if the women's singles winner is also in the mixed doubles winner
    if womens_singles_winner == mixed_doubles_winner:
        if womens_singles_winner in title_count:
            title_count[womens_singles_winner] += 1
        else:
            title_count[womens_singles_winner] = 1

# Find the champion with the most titles
max_titles = max(title_count.values())
most_champion = [name for name, count in title_count.items() if count == max_titles][0]

print(f"Final Answer: {most_champion}")