import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Calculate win, draw, loss percentages for each team
df['Total'] = df['Played']
df['Win%'] = (df['Won'] / df['Total']) * 100
df['Draw%'] = (df['Drawn'] / df['Total']) * 100
df['Loss%'] = (df['Lost'] / df['Total']) * 100

# Prepare data for plotting: index by Team, values by percentages
teams = df['Team']
win_percent = df['Win%']
draw_percent = df['Draw%']
loss_percent = df['Loss%']

# Create the stacked bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.25
index = range(len(teams))

# Plot each percentage as a segment
plt.bar(index, win_percent, bar_width, label='Wins', color='skyblue')
plt.bar(index, draw_percent, bar_width, label='Draws', color='lightgreen', bottom=win_percent)
plt.bar(index, loss_percent, bar_width, label='Losses', color='lightcoral', bottom=win_percent + draw_percent)

# Customize plot
plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentages by Team')
plt.xticks(index, teams, rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()