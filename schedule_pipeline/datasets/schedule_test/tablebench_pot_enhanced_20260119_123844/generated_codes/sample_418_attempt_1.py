import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'points per game' to float for analysis
df['points per game'] = pd.to_numeric(df['points per game'])

# Sort by tournament to ensure chronological order
df_sorted = df.sort_values(by='tournament')

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(df_sorted['tournament'], df_sorted['points per game'], marker='o', linestyle='-', color='b')
plt.title('Trend in Points Per Game Across Tournaments')
plt.xlabel('Tournament')
plt.ylabel('Points Per Game')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Identify significant changes (e.g., drop > 2 points or increase > 2 points)
trends = []
for i in range(1, len(df_sorted)):
    change = df_sorted['points per game'].iloc[i] - df_sorted['points per game'].iloc[i-1]
    if abs(change) >= 2:
        trends.append(f"{df_sorted['tournament'].iloc[i]}: {change:+.1f} points")

print(f"Final Answer: {', '.join(trends)}")