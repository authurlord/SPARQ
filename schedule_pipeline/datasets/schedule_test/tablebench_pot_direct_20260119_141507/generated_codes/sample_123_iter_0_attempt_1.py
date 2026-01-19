import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the relevant columns: Wins, Draws, Losses
wins = df['Wins'].astype(int)
draws = df['Draws'].astype(int)
losses = df['Losses'].astype(int)

# Create a list of club names
clubs = df['Club']

# Create a stacked bar chart
plt.figure(figsize=(14, 8))
bar_width = 0.25
index = range(len(clubs))

# Plot each category in a stacked manner
bars1 = plt.bar(index, wins, bar_width, label='Wins', color='skyblue')
bars2 = plt.bar(index, draws, bar_width, label='Draws', color='lightgreen', bottom=wins)
bars3 = plt.bar(index, losses, bar_width, label='Losses', color='lightcoral', bottom=wins + draws)

# Customize the chart
plt.xlabel('Club')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Match Statistics by Team')
plt.xticks(index, clubs, rotation=45)
plt.legend()

# Improve layout to prevent label cutoff
plt.tight_layout()

# Show the chart
plt.show()