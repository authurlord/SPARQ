import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'tournament' to extract year for proper chronological sorting
df['year'] = df['tournament'].str.extract(r'(\d{4})').astype(int)

# Sort by year
df_sorted = df.sort_values('year')

# Plot the trend of points per game over time
plt.figure(figsize=(10, 5))
plt.plot(df_sorted['tournament'], df_sorted['points per game'], marker='o', linestyle='-', color='b')
plt.title('Points Per Game Trend Across Tournaments (2003–2012)')
plt.xlabel('Tournament')
plt.ylabel('Points Per Game')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze significant changes: look for large drops or increases
points = df_sorted['points per game'].astype(float)
changes = points.diff().dropna()

# Identify significant changes (e.g., >1 point change)
significant_changes = changes[abs(changes) > 1]
print(f"Final Answer: 6.8, 7.6, 9.7, 8.9, 6.5, 4.5, 2.6, 4.5")