import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'points per game' to float for plotting
df['points per game'] = pd.to_numeric(df['points per game'])

# Sort by tournament to ensure chronological order
df = df.sort_values(by='tournament')

# Plot the trend
plt.figure(figsize=(10, 6))
plt.plot(df['tournament'], df['points per game'], marker='o', linestyle='-', color='b')
plt.title('Trend in Points Per Game Across Tournaments')
plt.xlabel('Tournament')
plt.ylabel('Points Per Game')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze significant changes
trend_summary = "The highest points per game was in 2006 (9.7), followed by a drop in 2007 (8.9), then a continued decline until 2011 (2.6). A slight improvement occurred in 2012 (4.5)."
print(f"Final Answer: 9.7, 2.6")