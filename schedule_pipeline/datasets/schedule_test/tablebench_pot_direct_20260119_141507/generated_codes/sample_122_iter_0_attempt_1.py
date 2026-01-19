import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Date' to datetime for proper plotting
df['Date'] = pd.to_datetime(df['Date'])

# Filter only "Win" results and create a cumulative count
df['Result'] = df['Result'].str.strip()
wins = df[df['Result'] == 'Win']

# Sort by Date to ensure chronological order
wins = wins.sort_values('Date')

# Compute cumulative wins
wins['Cumulative Wins'] = wins.groupby('Date').cumcount() + 1  # Each win adds one to cumulative

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(wins['Date'], wins['Cumulative Wins'], marker='o', linestyle='-', color='b')
plt.title('Cumulative Number of Wins Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Wins')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the final answer as requested
Final Answer: Cumulative Wins