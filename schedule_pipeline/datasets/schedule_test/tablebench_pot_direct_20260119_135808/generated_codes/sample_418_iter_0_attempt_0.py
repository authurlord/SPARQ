import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract year from tournament column for proper sorting
df['year'] = df['tournament'].str.extract(r'(\d{4})').astype(int)

# Sort by year
df = df.sort_values('year')

# Plot points per game over time
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['points per game'], marker='o', linestyle='-', color='b')
plt.title('Points Per Game Over Time')
plt.xlabel('Year')
plt.ylabel('Points Per Game')
plt.grid(True)
plt.xticks(df['year'], rotation=45)
plt.tight_layout()
plt.show()

# Identify significant changes (e.g., drop >2 points or increase >2 points)
changes = []
for i in range(1, len(df)):
    diff = float(df.iloc[i]['points per game']) - float(df.iloc[i-1]['points per game'])
    if abs(diff) >= 2:
        changes.append(f"{df.iloc[i]['year']} ({diff:+.1f})")

print(f"Final Answer: {', '.join(changes)}")