import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select top 8 teams (first 8 rows)
top_8 = df.head(8)

# Prepare data for stacked bar chart
teams = top_8['Club']
wins = top_8['Won']
draws = top_8['Drawn']
losses = top_8['Lost']

# Create the stacked bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(teams, wins, label='Won', color='skyblue')
plt.bar(teams, draws, bottom=wins, label='Drawn', color='lightgreen')
plt.bar(teams, losses, bottom=wins + draws, label='Lost', color='lightcoral')

plt.xlabel('Team')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses for Top 8 Ranked Teams')
plt.legend()
plt.tight_layout()

# Show the chart
plt.show()