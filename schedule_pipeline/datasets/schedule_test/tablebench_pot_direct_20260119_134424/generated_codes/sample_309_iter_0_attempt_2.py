import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the women's singles champions and mixed doubles winners
women_singles = df['womens singles'].tolist()
mixed_doubles = df['mixed doubles'].tolist()

# Dictionary to count how many times each women's singles champion also won mixed doubles
champion_count = {}

# Iterate through each year
for i, (singles, mixed) in enumerate(zip(women_singles, mixed_doubles)):
    # Skip if mixed doubles is 'no competition' or not a valid entry
    if mixed == 'no competition':
        continue
    # Split the mixed doubles winner into two names
    mixed_winners = [name.strip() for name in mixed.split(',')]
    # Check if the women's singles champion is in the mixed doubles winner
    if singles.split(' , ')[0] in mixed_winners:
        champion = singles.split(' , ')[0]
        champion_count[champion] = champion_count.get(champion, 0) + 1

# Find the champion with the maximum number of titles
if champion_count:
    most_titles_champion = max(champion_count, key=champion_count.get)
    print(f"Final Answer: {most_titles_champion}")
else:
    print("Final Answer: None")