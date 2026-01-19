import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'races' and 'wins' to numeric, handling 'n/a' and non-numeric values
df['races'] = pd.to_numeric(df['races'], errors='coerce')
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')

# Drop rows where races is 0 or NaN
df_filtered = df.dropna(subset=['races', 'wins'])
df_filtered = df_filtered[df_filtered['races'] > 0]

# Calculate win rate
df_filtered['win_rate'] = (df_filtered['wins'] / df_filtered['races']) * 100

# Find the series with the highest win rate
max_win_rate_series = df_filtered.loc[df_filtered['win_rate'].idxmax()]['series']

print(f"Final Answer: {max_win_rate_series}")