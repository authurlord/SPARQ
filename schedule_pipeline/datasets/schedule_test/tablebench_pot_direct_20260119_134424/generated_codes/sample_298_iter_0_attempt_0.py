import pandas as pd

df = pd.read_csv('table.csv')
# Filter for wins by The Washboard Union
wins = df[(df['Nominee/Work'] == 'The Washboard Union') & (df['Result'] == 'Won')]
# Count wins per award
award_counts = wins['Award'].value_counts()
# Get the award with the most wins
most_won_award = award_counts.idxmax()
# Find the first year they won this award
first_win_year = wins[wins['Award'] == most_won_award]['Year'].min()
print(f"Final Answer: {most_won_award}, {first_win_year}")