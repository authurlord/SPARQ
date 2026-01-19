import pandas as pd

df = pd.read_csv('table.csv')

# Find the team with the most titles
max_titles = df['winner (number of titles)'].str.extract(r'(\w+ \w+)', expand=False).dropna()
title_counts = df['winner (number of titles)'].str.count(r'\(')  # Count parentheses to get title count
# Parse the winner name and count
df['title_count'] = df['winner (number of titles)'].str.extract(r'\((\d+)\)').astype(float)
df['winner_name'] = df['winner (number of titles)'].str.extract(r'(\w+ \w+)').str.strip()

# Find the team with the maximum titles
max_title_team = df.loc[df['title_count'].idxmax(), 'winner_name']
first_win_year = df[df['winner_name'] == max_title_team].iloc[0]['year']

# Get the winning coach in the year of their first title
coach_in_first_win = df[df['year'] == first_win_year]['winning coach'].values[0]

print(f"Final Answer: {max_title_team}, {coach_in_first_win}")