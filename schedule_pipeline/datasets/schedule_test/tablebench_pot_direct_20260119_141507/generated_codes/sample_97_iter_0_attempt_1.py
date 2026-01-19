import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Select top 8 teams (rank 1 to 8)
top_8 = df.head(8)

# Prepare data for stacked bar chart
labels = top_8['Club']
wins = top_8['Won']
draws = top_8['Drawn']
losses = top_8['Lost']

# Create stacked bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, wins, label='Won', color='skyblue')
plt.bar(labels, draws, bottom=wins, label='Drawn', color='lightgreen')
plt.bar(labels, losses, bottom=wins + draws, label='Lost', color='lightcoral')

plt.xlabel('Club')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins, Draws, and Losses for Top 8 Teams')
plt.legend()
plt.tight_layout()
plt.show()

print("Final Answer: stacked_bar_chart")