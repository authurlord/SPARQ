import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'wins' and 'races' to numeric, coercing errors to NaN
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')
df['races'] = pd.to_numeric(df['races'], errors='coerce')

# Drop rows where either wins or races are NaN (invalid data)
df_clean = df.dropna(subset=['wins', 'races'])

# Group by 'series' and compute total wins and total races
grouped = df_clean.groupby('series').agg(total_wins=('wins', 'sum'), total_races=('races', 'sum')).reset_index()

# Calculate win rate as percentage
grouped['win_rate'] = (grouped['total_wins'] / grouped['total_races']) * 100

# Find the series with the highest win rate
max_win_rate_series = grouped.loc[grouped['win_rate'].idxmax(), 'series']

print(f"Final Answer: {max_win_rate_series}")