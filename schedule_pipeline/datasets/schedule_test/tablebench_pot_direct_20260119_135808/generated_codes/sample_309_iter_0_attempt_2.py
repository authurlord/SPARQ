import pandas as pd

df = pd.read_csv('table.csv')

# Extract the women's singles champions and mixed doubles champions
women_singles = df['womens singles']
mixed_doubles = df['mixed doubles']

# Dictionary to count wins for women who won both
win_count = {}

# Iterate through each row
for idx, row in df.iterrows():
    single_winner = row['womens singles'].split(', ')[0]  # First name only
    mixed_winners = row['mixed doubles'].split(' ')
    
    # Check if the women's singles champion is in the mixed doubles pair
    if 'mixed doubles' in row['mixed doubles'] and row['mixed doubles'] != 'no competition':
        # Split the mixed doubles string to extract names
        mixed_names = row['mixed doubles'].split(' ')
        # Look for the woman's name (likely in the second part or last)
        # The pattern seems to be: [man] [woman]
        # So the last name in the string is likely the woman
        if len(mixed_names) >= 2:
            woman_in_mixed = mixed_names[-1]
            # Clean up potential punctuation
            woman_in_mixed = woman_in_mixed.strip(',.').rstrip(' ,')
            if woman_in_mixed == single_winner.split(' ')[-1]:
                win_count[single_winner] = win_count.get(single_winner, 0) + 1
            elif woman_in_mixed == single_winner:
                win_count[single_winner] = win_count.get(single_winner, 0) + 1

# Find the woman with the most titles
if win_count:
    max_wins = max(win_count.values())
    most_champions = [name for name, count in win_count.items() if count == max_wins]
    final_answer = most_champions[0]
else:
    final_answer = "No data"

print(f"Final Answer: {final_answer}")