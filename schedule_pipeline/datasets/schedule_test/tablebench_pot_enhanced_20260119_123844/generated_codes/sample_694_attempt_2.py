import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum wins
max_wins_row = df.loc[df['wins'].idxmax()]
# Extract the season
season_with_max_wins = max_wins_row['season']
print(f"Final Answer: {season_with_max_wins}")