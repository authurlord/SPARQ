import pandas as pd

df = pd.read_csv('table.csv')
# Remove rows where races is 0 to avoid division by zero
df = df[df['races'] > 0]
# Calculate win rate
df['win_rate'] = (df['wins'] / df['races']) * 100
# Find the series with the highest win rate
max_win_rate_series = df.loc[df['win_rate'].idxmax()]['series']
print(f"Final Answer: {max_win_rate_series}")