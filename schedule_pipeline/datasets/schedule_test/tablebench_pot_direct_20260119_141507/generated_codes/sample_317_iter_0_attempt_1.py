import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where winning team is "united states"
us_wins = df[df['winning team'] == 'united states']
# Group by 'us captain' and count the number of wins
captain_counts = us_wins['us captain'].value_counts()
# Find the maximum count
max_wins = captain_counts.max()
print(f"Final Answer: {max_wins}")