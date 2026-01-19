import pandas as pd

df = pd.read_csv('table.csv')
# Calculate win rate for each series
df['win_rate'] = (df['wins'] / df['races']) * 100
# Find the series with the highest win rate
max_win_rate_series = df.loc[df['win_rate'].idxmax(), 'series']
print(f"Final Answer: {max_win_rate_series}")