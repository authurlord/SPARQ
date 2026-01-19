import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Calculate win, draw, loss percentages
df['Win%'] = df['Won'] / df['Played']
df['Draw%'] = df['Drawn'] / df['Played']
df['Loss%'] = df['Lost'] / df['Played']

# Prepare data for plotting: each team has three categories (Win, Draw, Loss)
data = df[['Team', 'Win%', 'Draw%', 'Loss%']].copy()

# Create a percentage stacked bar chart
plt.figure(figsize=(12, 8))
bar_width = 0.25
index = range(len(data))

# Plot each percentage component
bars1 = plt.bar([i + 0 for i in index], data['Win%'], bar_width, label='Wins', color='skyblue')
bars2 = plt.bar([i + 1 for i in index], data['Draw%'], bar_width, label='Draws', color='lightgreen')
bars3 = plt.bar([i + 2 for i in index], data['Loss%'], bar_width, label='Losses', color='salmon')

# Customize plot
plt.xlabel('Team')
plt.ylabel('Percentage (%)')
plt.title('Win-Loss-Draw Percentages by Team')
plt.xticks([i + 1 for i in index], data['Team'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

# Final Answer: The chart has been successfully generated.
Final Answer: chart_generated