import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Result' column to binary: 1 for Win, 0 otherwise
df['Win'] = df['Result'].apply(lambda x: 1 if x == 'Win' else 0)

# Calculate cumulative sum of wins
df['Cumulative Wins'] = df['Win'].cumsum()

# Sort by Date to ensure chronological order
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Cumulative Wins'], marker='o', linestyle='-', color='b')
plt.title('Cumulative Number of Wins Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Wins')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the final cumulative win values as per the requirement
print(f"Final Answer: {', '.join(map(str, df['Cumulative Wins'].tolist()))}")