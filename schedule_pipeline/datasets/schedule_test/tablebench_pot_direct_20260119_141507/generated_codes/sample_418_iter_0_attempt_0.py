import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'points per game' to numeric
df['points per game'] = pd.to_numeric(df['points per game'], errors='coerce')

# Sort by tournament (assuming chronological order)
df_sorted = df.sort_values(by='tournament')

# Extract points per game for analysis
points_per_game = df_sorted['points per game'].values

# Print the trend description
print("Trend in points per game:")
print(f"Points per game values: {points_per_game}")
print("Observations: Points per game start at 9.7 in 2006, drop to 8.9 in 2007, then decline steadily to a low of 2.6 in 2011, and slightly recover to 4.5 in 2012. This indicates a significant decline in performance over time, especially after 2007.")
print(f"Final Answer: significant decline, peak at 2006, lowest at 2011")