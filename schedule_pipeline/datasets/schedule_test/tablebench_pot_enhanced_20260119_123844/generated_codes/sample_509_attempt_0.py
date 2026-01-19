import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'races' and 'wins' to numeric, coercing errors to NaN
df['races'] = pd.to_numeric(df['races'], errors='coerce')
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')

# Drop rows where races or wins are NaN
df = df.dropna(subset=['races', 'wins'])

# Calculate win rate
df['win_rate'] = (df['wins'] / df['races']) * 100

# Find the series with the highest win rate
max_win_rate_series = df.loc[df['win_rate'].idxmax()]['series']

print(f"Final Answer: {max_win_rate_series}")