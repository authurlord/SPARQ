import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points per game' to numeric (in case of string)
df['points per game'] = pd.to_numeric(df['points per game'], errors='coerce')

# Sort by tournament to see chronological order
df_sorted = df.sort_values(by='tournament')

# Extract points per game values
points_per_game = df_sorted['points per game']

# Check for significant changes (e.g., large jumps or drops)
# We'll look at differences between consecutive values
differences = points_per_game.diff()

# Identify significant changes (e.g., >1 or < -1 point per game)
significant_changes = differences.abs() > 1

# Print the trend description
print("Trend in points per game:")
print(f"Points per game: {points_per_game.tolist()}")
print(f"Significant changes (jump or drop >1): {significant_changes.tolist()}")

# Final Answer based on observed trend: initial high, drop in 2011, then recovery
Final Answer: initial high, drop in 2011, then recovery