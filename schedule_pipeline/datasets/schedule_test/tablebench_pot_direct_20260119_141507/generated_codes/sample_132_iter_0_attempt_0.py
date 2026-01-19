import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert the data into percentages for Won, Drawn, Lost
df['Total'] = df['Played']
df['Win%'] = (df['Won'] / df['Total']) * 100
df['Draw%'] = (df['Drawn'] / df['Total']) * 100
df['Loss%'] = (df['Lost'] / df['Total']) * 100

# Prepare data for plotting
teams = df['Team']
win_percent = df['Win%']
draw_percent = df['Draw%']
loss_percent = df['Loss%']

# Create the stacked bar chart
plt.figure(figsize=(12, 7))
bar_width = 0.25
index = range(len(teams))

plt.bar(index, win_percent, bar_width, label='Wins', color='skyblue')
plt.bar([i + bar_width for i in index], draw_percent, bar_width, label='Draws', color='lightgreen')
plt.bar([i + 2*bar_width for i in index], loss_percent, bar_width, label='Losses', color='salmon')

plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentages by Team')
plt.xticks([i + bar_width for i in index], teams, rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer is not a value but a chart, so we just print a placeholder indicating completion
Final Answer: chart_generated