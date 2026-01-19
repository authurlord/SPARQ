import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'points per game' to float for analysis
df['points per game'] = pd.to_numeric(df['points per game'])

# Extract year from tournament column to sort chronologically
df['year'] = df['tournament'].str.extract(r'(\d{4})').astype(int)

# Sort by year
df = df.sort_values('year')

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

# Analyze significant changes: look for large drops or rises
trends = []
for i in range(1, len(df)):
    change = df['points per game'].iloc[i] - df['points per game'].iloc[i-1]
    if abs(change) > 2:
        trends.append(f"{df['tournament'].iloc[i-1]} to {df['tournament'].iloc[i]}: {change:+.1f}")

print("Significant changes in performance:", ", ".join(trends))

# Final answer: summarize key trend (highest and lowest points per game)
max_ppg = df['points per game'].max()
min_ppg = df['points per game'].min()
final_answer = f"{max_ppg}, {min_ppg}"
print(f"Final Answer: {final_answer}")