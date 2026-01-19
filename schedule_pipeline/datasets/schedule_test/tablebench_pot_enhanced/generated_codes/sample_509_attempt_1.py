import pandas as pd

df = pd.read_csv('table.csv')

# Drop rows where races or wins are 0 to avoid division by zero
df_filtered = df[df['races'] > 0]

# Calculate win rate
df_filtered['win_rate'] = (df_filtered['wins'] / df_filtered['races']) * 100

# Find the series with the highest win rate
max_win_rate_series = df_filtered.loc[df_filtered['win_rate'].idxmax()]['series']

print(f"Final Answer: {max_win_rate_series}")