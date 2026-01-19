import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'points per game' to float for analysis
df['points per game'] = pd.to_numeric(df['points per game'])

# Sort by tournament chronologically (assuming order is chronological)
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

# Identify significant changes (e.g., drop from 9.7 to 6.5, then to 4.5)
# Key observation: Performance peaked in 2006 (9.7) and declined sharply after 2009 (6.5 → 4.5)

print("Final Answer: 9.7, 4.5")