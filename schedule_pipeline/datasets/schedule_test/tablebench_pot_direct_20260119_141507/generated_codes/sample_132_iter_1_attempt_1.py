import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Calculate win, draw, loss percentages for each team
df['Win%'] = df['Won'] / df['Played'] * 100
df['Draw%'] = df['Drawn'] / df['Played'] * 100
df['Loss%'] = df['Lost'] / df['Played'] * 100

# Prepare data for plotting: team names and their percentages
teams = df['Team']
win_percentages = df['Win%']
draw_percentages = df['Draw%']
loss_percentages = df['Loss%']

# Create the stacked bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(teams, win_percentages, label='Wins', color='skyblue')
plt.bar(teams, draw_percentages, bottom=win_percentages, label='Draws', color='lightgreen')
plt.bar(teams, loss_percentages, bottom=win_percentages + draw_percentages, label='Losses', color='lightcoral')

# Customize the chart
plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentages by Team')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Display the chart
plt.show()

# Final Answer: The chart has been generated successfully.
Final Answer: chart_generated