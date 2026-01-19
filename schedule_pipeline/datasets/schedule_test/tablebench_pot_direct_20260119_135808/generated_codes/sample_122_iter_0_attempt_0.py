import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date to ensure chronological order
df = df.sort_values('Date')

# Create a cumulative sum of wins
df['Win'] = (df['Result'] == 'Win').astype(int)
df['Cumulative Wins'] = df['Win'].cumsum()

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Cumulative Wins'], marker='o', linestyle='-', color='blue')
plt.title('Cumulative Number of Wins Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Wins')
plt.grid(True)
plt.tight_layout()
plt.show()