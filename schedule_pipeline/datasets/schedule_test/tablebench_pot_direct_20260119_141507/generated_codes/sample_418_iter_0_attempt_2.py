import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points per game' to numeric (in case of string)
df['points per game'] = pd.to_numeric(df['points per game'], errors='coerce')

# Sort by tournament to analyze chronological trend
df_sorted = df.sort_values(by='tournament')

# Extract points per game values and look at trends
points_per_game = df_sorted['points per game'].values

# Identify significant changes (e.g., jumps or drops)
changes = []
for i in range(1, len(points_per_game)):
    diff = points_per_game[i] - points_per_game[i-1]
    if abs(diff) > 1.0:  # Significant change threshold
        changes.append((df_sorted.iloc[i-1]['tournament'], df_sorted.iloc[i]['tournament'], diff))

print(f"Final Answer: {changes}")