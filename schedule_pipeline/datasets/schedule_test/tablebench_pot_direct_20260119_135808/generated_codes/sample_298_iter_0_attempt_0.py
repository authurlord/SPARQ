import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the nominee/work is 'The Washboard Union' and result is 'Won'
wins = df[(df['Nominee/Work'] == 'The Washboard Union') & (df['Result'] == 'Won')]

# Count wins per award
award_counts = wins['Award'].value_counts()

# Get the award with the most wins
most_wins_award = award_counts.idxmax()
most_wins_count = award_counts.max()

# Find the first year they won this award
first_win_year = wins[wins['Award'] == most_wins_award]['Year'].min()

print(f"Final Answer: {most_wins_award}, {first_win_year}")