import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum wins
max_wins_row = df.loc[df['wins'].idxmax()]
# Extract the season with the highest wins
highest_wins_season = max_wins_row['season']
print(f"Final Answer: {highest_wins_season}")