import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric (in case of string or non-numeric)
df['Won'] = pd.to_numeric(df['Won'], errors='coerce')
df['Lost'] = pd.to_numeric(df['Lost'], errors='coerce')
df['Drawn'] = pd.to_numeric(df['Drawn'], errors='coerce')
df['Played'] = pd.to_numeric(df['Played'], errors='coerce')

# Calculate percentages
df['Win%'] = df['Won'] / df['Played'] * 100
df['Loss%'] = df['Lost'] / df['Played'] * 100
df['Draw%'] = df['Drawn'] / df['Played'] * 100

# Prepare data for plotting
teams = df['Team']
win_percentages = df['Win%']
loss_percentages = df['Loss%']
draw_percentages = df['Draw%']

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Create stacked bar chart
bar_width = 0.25
index = range(len(teams))
bars1 = plt.bar(index, win_percentages, bar_width, label='Wins', color='skyblue')
bars2 = plt.bar(index, loss_percentages, bar_width, label='Losses', bottom=win_percentages, color='salmon')
bars3 = plt.bar(index, draw_percentages, bar_width, label='Draws', bottom=win_percentages + loss_percentages, color='lightgreen')

# Add labels and title
plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentage by Team')
plt.xticks(index, teams, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer is not a number or entity, but the chart is generated as required.
Final Answer: chart_generated