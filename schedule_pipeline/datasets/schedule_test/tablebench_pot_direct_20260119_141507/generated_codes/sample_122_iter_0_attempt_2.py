import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter only the 'Win' results
wins_df = df[df['Result'] == 'Win']

# Sort by date and group to get cumulative wins
wins_df = wins_df.sort_values('Date')
wins_df['Cumulative_Wins'] = wins_df.groupby('Date').cumcount() + 1

# Create a list of dates and cumulative wins
dates = wins_df['Date'].dt.date.tolist()
cumulative_wins = wins_df['Cumulative_Wins'].tolist()

# Plot the line chart
plt.figure(figsize=(12, 6))
plt.plot(dates, cumulative_wins, marker='o', linestyle='-', color='b')
plt.title('Cumulative Number of Wins Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Wins')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Since we are plotting, no final answer in numeric form is required
# But if needed, we can print the last cumulative win as a summary
print(f"Final Answer: {cumulative_wins[-1]}")