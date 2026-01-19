import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Select top 8 teams
top_8_teams = df.head(8)
# Set the index to Club for better labeling
top_8_teams.set_index('Club', inplace=True)
# Create a stacked bar chart
top_8_teams[['Won', 'Drawn', 'Lost']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Wins, Draws, and Losses for Top 8 Ranked Teams')
plt.xlabel('Club')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.legend(title='Result')
plt.tight_layout()
plt.show()