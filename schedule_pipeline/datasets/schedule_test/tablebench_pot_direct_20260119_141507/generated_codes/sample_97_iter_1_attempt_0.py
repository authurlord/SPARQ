import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select top 8 teams (first 8 rows)
top_8 = df.head(8)

# Prepare data for plotting: wins, draws, losses
teams = top_8['Club']
wins = top_8['Won']
draws = top_8['Drawn']
losses = top_8['Lost']

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = range(len(teams))

# Plot each category on top of each other
plt.bar(index, wins, bar_width, label='Wins', color='skyblue')
plt.bar(index, draws, bar_width, label='Draws', color='lightgreen', bottom=wins)
plt.bar(index, losses, bar_width, label='Losses', color='lightcoral', bottom=wins + draws)

# Labels and title
plt.xlabel('Team')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses for Top 8 Teams')
plt.xticks(index, teams, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()