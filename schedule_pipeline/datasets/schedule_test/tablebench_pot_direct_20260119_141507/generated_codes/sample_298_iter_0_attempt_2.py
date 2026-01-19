import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Nominee/Work is 'The Washboard Union' and Result is 'Won'
wins = df[(df['Nominee/Work'] == 'The Washboard Union') & (df['Result'] == 'Won')]

# Group by Award and count wins
award_counts = wins.groupby('Award').size().reset_index(name='win_count')

# Find the award with the highest win count
most_wins_award = award_counts.loc[award_counts['win_count'].idxmax(), 'Award']

# Find the first year they won that award
first_win_year = wins[wins['Award'] == most_wins_award]['Year'].min()

print(f"Final Answer: {most_wins_award}, {first_win_year}")