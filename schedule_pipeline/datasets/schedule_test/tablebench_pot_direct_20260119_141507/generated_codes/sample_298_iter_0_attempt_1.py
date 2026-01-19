import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Nominee/Work is 'The Washboard Union' and Result is 'Won'
won_rows = df[(df['Nominee/Work'] == 'The Washboard Union') & (df['Result'] == 'Won')]

# Group by Award and count wins
award_counts = won_rows['Award'].value_counts()

# Get the award with the highest number of wins
most_won_award = award_counts.idxmax()

# Find the first year they won this award
first_win_year = won_rows[won_rows['Award'] == most_won_award]['Year'].min()

print(f"Final Answer: {most_won_award}, {first_win_year}")