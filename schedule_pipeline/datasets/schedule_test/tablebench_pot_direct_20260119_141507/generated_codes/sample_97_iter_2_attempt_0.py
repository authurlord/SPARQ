import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select top 8 teams (rank 1 to 8)
top_8 = df[df['rank'].astype(int).le(8)]

# Extract the relevant columns
wins = top_8['Won'].astype(int)
draws = top_8['Drawn'].astype(int)
losses = top_8['Lost'].astype(int)

# Create a list of team names (Club) for x-axis
clubs = top_8['Club'].values

# Create the stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = range(len(clubs))

# Plot wins, draws, and losses as stacked bars
plt.bar(index, wins, bar_width, label='Wins', color='skyblue')
plt.bar([i + bar_width for i in index], draws, bar_width, label='Draws', color='lightgreen')
plt.bar([i + 2*bar_width for i in index], losses, bar_width, label='Losses', color='salmon')

# Customize the plot
plt.xlabel('Teams')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart: Wins, Draws, and Losses for Top 8 Ranked Teams')
plt.xticks([i + bar_width for i in index], clubs, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Display the chart
plt.show()

# Final Answer: The chart has been generated successfully.
Final Answer: chart_generated