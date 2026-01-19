import pandas as pd

df = pd.read_csv('table.csv')

# Create a list to store players who won both men's singles and mixed doubles in the same year
players_with_both_titles = []

# Iterate through each row (year)
for index, row in df.iterrows():
    mens_singles = row['mens singles']
    mixed_doubles = row['mixed doubles']
    
    # Check if the same player won both titles in the same year
    if mens_singles in mixed_doubles:
        players_with_both_titles.append(mens_singles)

# Count occurrences of each player winning both titles in the same year
from collections import Counter
count = Counter(players_with_both_titles)

# Find players who won both titles at least twice
players_meeting_criteria = [player for player, freq in count.items() if freq >= 2]

# Print the result
print(f"Final Answer: {players_meeting_criteria[0]}")