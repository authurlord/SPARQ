import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select top 8 teams (first 8 rows)
top_8_teams = df.head(8)

# Extract wins, draws, losses
wins = top_8_teams['Won'].astype(int)
draws = top_8_teams['Drawn'].astype(int)
losses = top_8_teams['Lost'].astype(int)

# Create a list of team names (Club column)
teams = top_8_teams['Club'].values

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = range(len(teams))

# Plot wins, draws, losses as stacked bars
bars1 = plt.bar(index, wins, bar_width, label='Wins', color='skyblue')
bars2 = plt.bar(index, draws, bar_width, label='Draws', color='lightgreen', bottom=wins)
bars3 = plt.bar(index, losses, bar_width, label='Losses', color='salmon', bottom=wins + draws)

# Labels and title
plt.xlabel('Teams')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses for Top 8 Ranked Teams')
plt.xticks(index, teams, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer: The chart has been generated as requested.
Final Answer: chart_generated