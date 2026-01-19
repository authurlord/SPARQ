import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'races' and 'wins' to numeric, coercing errors to NaN
df['races'] = pd.to_numeric(df['races'], errors='coerce')
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')

# Drop rows where races or wins are NaN
df_clean = df.dropna(subset=['races', 'wins'])

# Calculate win rate
df_clean['win_rate'] = (df_clean['wins'] / df_clean['races']) * 100

# Find the series with the highest win rate
max_win_rate_series = df_clean.loc[df_clean['win_rate'].idxmax(), 'series']

print(f"Final Answer: {max_win_rate_series}")