import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'womens singles' and 'mixed doubles' columns
women_singles = df['womens singles']
mixed_doubles = df['mixed doubles']

# Create a dictionary to count wins for women's singles players who also won mixed doubles
singles_and_mixed = {}

# Iterate through each row
for idx, row in df.iterrows():
    singles_winner = row['womens singles'].split(',')[0].strip()  # Get the player name only
    mixed_winner = row['mixed doubles'].split()[0]  # First name from mixed doubles winner
    
    # Check if the woman won mixed doubles (if she is listed as the first player)
    if singles_winner == mixed_winner:
        singles_and_mixed[singles_winner] = singles_and_mixed.get(singles_winner, 0) + 1

# Find the champion with the most titles among those who won both
if singles_and_mixed:
    most_titles = max(singles_and_mixed, key=singles_and_mixed.get)
    print(f"Final Answer: {most_titles}")
else:
    print("Final Answer: None")